"""graphdb — direct push of a graphify graph into Neo4j / FalkorDB.

The FalkorDB writer goes through ``GraphStore`` (``graphify/store.py``), the
same batched writer graphify uses for its own graphs, so a pushed graph gets:

  - the shared ``:Entity`` label plus its file-type label (``:Entity:Python``),
    which is what makes a pushed graph readable by ``graphify query`` / ``serve``
    — they match on ``:Entity`` and saw nothing at all in a pushed graph before;
  - the ``n.id`` index, so edge-endpoint MATCHes resolve through an index
    instead of scanning every node once per edge (#2258);
  - batched ``UNWIND`` writes rather than one round trip per node and per edge.

Convergence (#3057). By default a push only adds and updates, so anything the
source has since pruned survives in the target forever and the two silently
diverge. ``prune=True`` makes the push *converge*: every node this push did not
write is deleted — nodes *and* edges, so an edge dropped between two surviving
endpoints goes too — and the target ends up an exact mirror of the source. Because
that is destructive it is opt-in, and it refuses to run when the deletion would
exceed ``shrink_limit`` of the target unless ``allow_shrink=True`` — the same
"refuse to SILENTLY drop nodes" rule as the #479 build guard.

Pruning deletes by *absence from this push*, not by repo. Point a push at a
graph holding anything you did not push and ``prune=True`` will remove it; use
``graph_name`` to give each source its own target graph.

The Neo4j writer is deliberately untouched: it has the same per-row and
unindexed-MATCH problems, but neither the fix nor a convergence mode can be
tested here, so it keeps its old add-only behavior and the new flags are
refused rather than silently ignored on that path.
"""
from __future__ import annotations

import re
import time

from graphify.analyze import _node_community_map

# Rows per UNWIND batch. Matches GraphStore._BATCH so both writers behave the
# same way against the same server.
_BATCH = 1000
# Nodes deleted per convergence page. Paged so a large prune is not one
# unbounded transaction.
_DELETE_PAGE = 10_000
# Refuse a prune that would delete more than this fraction of the target.
_DEFAULT_SHRINK_LIMIT = 0.20
# Stamped on every node a push writes; convergence deletes whatever lacks the
# current value. A plain (non-underscore) name so GraphStore._scalar_props keeps
# it — underscore-prefixed properties are dropped on the way into the store.
_EPOCH_PROP = "graphify_push_epoch"


def _safe_rel(relation: str) -> str:
    return re.sub(r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_").replace("-", "_")) or "RELATED_TO"


def _safe_label(label: str) -> str:
    """Sanitize a node label to prevent Cypher injection."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)
    return sanitized if sanitized else "Entity"


def _new_epoch() -> int:
    """Identifier for one push. Millisecond clock: two pushes into the same
    target never collide, and the value is meaningful when read back."""
    return int(time.time() * 1000)


def _chunked(iterable, size: int):
    """Yield lists of at most `size` items, holding only one chunk at a time.

    The source is a streaming GraphStore view on this branch, so the push must
    never materialize the whole graph — that is the OOM the reporter of #3057
    hit on a 1.87GB graph.
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _stamped_nodes(G, node_community: dict, epoch: int):
    """Yield (node_id, attrs) with community and the push epoch merged in."""
    for node_id, data in G.nodes(data=True):
        attrs = dict(data)
        cid = node_community.get(node_id)
        if cid is not None:
            attrs["community"] = cid
        attrs[_EPOCH_PROP] = epoch
        yield (node_id, attrs)


def _stamped_edges(G, epoch: int):
    """Yield (u, v, attrs) with the push epoch merged in, so the convergence
    sweep can tell this push's edges from ones the source has since dropped."""
    for u, v, data in G.edges(data=True):
        attrs = dict(data)
        attrs[_EPOCH_PROP] = epoch
        yield (u, v, attrs)


# A node this push did not write, and an edge this push did not write. Edges
# need their own sweep: DETACH DELETE on stale nodes takes their edges with
# them, but an edge dropped from the source whose two endpoints both survive
# would otherwise linger forever — the surplus-edge half of #3057.
_STALE_NODES = f"MATCH (n:Entity) WHERE n.{_EPOCH_PROP} IS NULL OR n.{_EPOCH_PROP} <> $epoch"
_STALE_EDGES = (
    f"MATCH (:Entity)-[r]->(:Entity) "
    f"WHERE r.{_EPOCH_PROP} IS NULL OR r.{_EPOCH_PROP} <> $epoch"
)


def _delete_paged(run, count, stale_match: str, var: str, params: dict, expected: int) -> int:
    """Delete `stale_match` in pages until none remain. Returns rows deleted.

    LIMIT does not page a DELETE in FalkorDB: its known-limitations doc notes
    LIMIT "does not currently short-circuit eager operations like CREATE, SET,
    or DELETE", so `... DELETE n LIMIT $page` deletes everything matched rather
    than a page. The LIMIT has to sit in a WITH that precedes the DELETE, as
    below.
    """
    verb = "DETACH DELETE" if var == "n" else "DELETE"
    page = f"{stale_match} WITH {var} LIMIT {_DELETE_PAGE} {verb} {var}"
    remaining = expected
    while remaining > 0:
        run(page, params)
        after = count(f"{stale_match} RETURN count({var})", params)
        if after >= remaining:
            raise RuntimeError(
                f"graphify: prune stalled with {after} stale rows remaining (no "
                f"progress in one page). Target may be read-only, or the delete "
                f"may be racing another writer."
            )
        remaining = after
    return expected - remaining


# ---------------------------------------------------------------------------
# Repo-keyed delta
#
# `global add` already treats the repo as the unit of change: it prunes a repo
# whole, re-adds it whole, records a per-repo source_hash in the global
# manifest, and returns skipped=True when that hash has not moved. The delta
# push mirrors that contract instead of inventing one — only repos whose hash
# moved are re-sent, so a 226-repo global graph with one changed repo sends one
# repo's rows rather than all of them (#3057).
#
# The "what did I last push" state lives in the TARGET database, not in a local
# ledger, as a :GraphifyPushState node per repo. A ledger cannot notice that the
# database was wiped or that a run half-landed — it still reads clean and the
# delta never repairs the drift. Reading the target's own per-repo node counts
# and re-pushing any repo whose count disagrees with the manifest turns silent
# permanent drift into automatic repair.
_STATE_LABEL = "GraphifyPushState"


def _read_push_state(count_rows) -> dict[str, dict]:
    """Per-repo {source_hash, node_count} the target believes it holds."""
    rows = count_rows(
        f"MATCH (s:{_STATE_LABEL}) RETURN s.repo, s.source_hash, s.node_count", {}
    )
    return {r[0]: {"source_hash": r[1], "node_count": int(r[2] or 0)} for r in rows if r[0]}


def _target_repo_counts(count_rows) -> dict[str, int]:
    """The target's OWN per-repo node counts — one indexed aggregate. This is
    the check a ledger cannot do: it sees a wipe or a half-landed run."""
    rows = count_rows(
        "MATCH (n:Entity) WHERE n.repo IS NOT NULL RETURN n.repo, count(n)", {}
    )
    return {r[0]: int(r[1]) for r in rows if r[0]}


def _repo_nodes(G, tag: str):
    """Stream one repo's nodes from the source global graph."""
    from graphify.store import _META_KEY

    for r in G._stream(
        "MATCH (n:Entity {repo:$t}) RETURN n.id, properties(n)", "id(n)", {"t": tag}
    ):
        attrs = dict(r[1])
        attrs.pop(_META_KEY, None)
        yield (r[0], attrs)


def _repo_edges(G, tag: str):
    """Stream every edge incident to one repo's nodes — in EITHER direction.

    Not just edges whose both endpoints are in the repo: `global add` remaps
    external-library nodes onto whichever repo first contributed them, so a
    cross-repo edge B->A is owned by neither B nor A alone. Pruning repo A drops
    that edge with A's node; re-adding only A's own edges would not bring it
    back, and the target would quietly lose cross-repo connectivity on every
    delta. Selecting on `a.repo = t OR b.repo = t` restores it, and repo B's
    nodes are never touched.
    """
    for r in G._stream(
        "MATCH (a:Entity)-[r]->(b:Entity) WHERE a.repo = $t OR b.repo = $t "
        "RETURN a.id, b.id, properties(r)", "id(r)", {"t": tag},
    ):
        yield (r[0], r[1], dict(r[2]))


def _prune_repo_paged(run, count, tag: str) -> int:
    """Delete one repo's nodes, paged. Returns the count removed.

    Not GraphStore.prune_repo: that measures with two whole-graph
    number_of_nodes() calls per repo, which on a 226-repo global graph is 452
    full scans. One scoped count over the indexed `repo` property does the same
    job per repo.
    """
    scoped = "MATCH (n:Entity {repo: $t})"
    params = {"t": tag}
    n = count(f"{scoped} RETURN count(n)", params)
    if n:
        _delete_paged(run, count, scoped, "n", params, n)
    return n


def _plan_delta(manifest_repos: dict, state: dict, live_counts: dict) -> tuple[list, list, dict]:
    """Decide which repos to re-push and which to delete.

    Returns (changed, removed, reasons). A repo is re-pushed when its manifest
    hash moved, when the target never recorded it, or when the target's live
    node count disagrees with what the manifest says it should hold (drift
    repair). A repo the manifest no longer lists is deleted.
    """
    changed, reasons = [], {}
    for tag, info in manifest_repos.items():
        want_hash = info.get("source_hash")
        want_count = int(info.get("node_count") or 0)
        known = state.get(tag)
        live = live_counts.get(tag, 0)
        if known is None:
            changed.append(tag); reasons[tag] = "not present in target"
        elif known.get("source_hash") != want_hash:
            changed.append(tag); reasons[tag] = "source changed"
        elif live != want_count:
            changed.append(tag)
            reasons[tag] = f"target drift ({live} nodes in target, manifest says {want_count})"
    removed = [t for t in list(state) + list(live_counts) if t not in manifest_repos]
    # de-dup while keeping order
    removed = list(dict.fromkeys(removed))
    return changed, removed, reasons


def _converge(run, count, epoch: int, allow_shrink: bool, shrink_limit: float) -> tuple[int, int]:
    """Delete every node and edge this push did not write.

    Returns (nodes_deleted, edges_deleted). `run(cypher, params)` executes;
    `count(cypher, params)` returns an int.
    """
    params = {"epoch": epoch}
    stale_n = count(f"{_STALE_NODES} RETURN count(n)", params)
    stale_e = count(f"{_STALE_EDGES} RETURN count(r)", params)
    if stale_n <= 0 and stale_e <= 0:
        return 0, 0

    total_n = count("MATCH (n:Entity) RETURN count(n)", {})
    total_e = count("MATCH (:Entity)-[r]->(:Entity) RETURN count(r)", {})
    if not allow_shrink:
        for kind, stale, total in (("nodes", stale_n, total_n), ("edges", stale_e, total_e)):
            if total > 0 and (stale / total) > shrink_limit:
                raise ValueError(
                    f"graphify: push --prune would delete {stale} of {total} "
                    f"{kind} ({stale / total:.0%}) from the target graph, over "
                    f"the {shrink_limit:.0%} safety limit. That usually means "
                    f"the push is aimed at the wrong graph — check --graph-name. "
                    f"Pass --allow-shrink if the removal is intended. Nothing "
                    f"was deleted; the additive part of this push has already "
                    f"been applied."
                )

    # Nodes first: DETACH DELETE takes their edges with them, so the edge sweep
    # that follows has less to do and its count is already settled.
    nodes_deleted = _delete_paged(run, count, _STALE_NODES, "n", params, stale_n) if stale_n else 0
    remaining_e = count(f"{_STALE_EDGES} RETURN count(r)", params)
    edges_deleted = (
        _delete_paged(run, count, _STALE_EDGES, "r", params, remaining_e) if remaining_e else 0
    )
    return nodes_deleted, edges_deleted


def push_to_neo4j(
    G: nx.Graph,
    uri: str,
    user: str,
    password: str,
    communities: dict[int, list[str]] | None = None,
) -> dict[str, int]:
    """Push graph directly to a running Neo4j instance via the Python driver.

    Requires: pip install neo4j

    Uses MERGE so re-running is safe - nodes and edges are upserted, not duplicated.
    Returns a dict with counts of nodes and edges pushed.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j driver not installed. Run: pip install neo4j"
        ) from e

    node_community = _node_community_map(communities) if communities else {}

    def _safe_rel(relation: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "_", relation.upper().replace(" ", "_").replace("-", "_")) or "RELATED_TO"

    def _safe_label(label: str) -> str:
        """Sanitize a Neo4j node label to prevent Cypher injection."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)
        return sanitized if sanitized else "Entity"

    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes_pushed = 0
    edges_pushed = 0

    with driver.session() as session:
        for node_id, data in G.nodes(data=True):
            props = {
                k: v for k, v in data.items()
                if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
            }
            props["id"] = node_id
            cid = node_community.get(node_id)
            if cid is not None:
                props["community"] = cid
            ftype = _safe_label(data.get("file_type", "Entity").capitalize())
            session.run(
                f"MERGE (n:{ftype} {{id: $id}}) SET n += $props",
                id=node_id,
                props=props,
            )
            nodes_pushed += 1

        for u, v, data in G.edges(data=True):
            rel = _safe_rel(data.get("relation", "RELATED_TO"))
            props = {
                k: v for k, v in data.items()
                if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
            }
            session.run(
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r += $props",
                src=u,
                tgt=v,
                props=props,
            )
            edges_pushed += 1

    driver.close()
    return {"nodes": nodes_pushed, "edges": edges_pushed}


def _push_delta(
    G, target, node_community: dict, epoch: int, manifest_repos: dict,
    run, count, rows, allow_shrink: bool, shrink_limit: float,
) -> dict:
    """Repo-keyed delta push. See the module's delta section for the contract."""
    state = _read_push_state(rows)
    live_counts = _target_repo_counts(rows)
    changed, removed, reasons = _plan_delta(manifest_repos, state, live_counts)

    # Size guard before anything is deleted: a manifest that does not belong to
    # this database looks exactly like a genuine mass removal. Same rule as the
    # #479 build guard, applied to the push.
    #
    # Only NET removal counts. A re-pushed repo is pruned and immediately
    # re-added, so its nodes are not lost — charging them to the limit would
    # refuse any delta touching more than shrink_limit of a small global graph.
    # A repo that comes back SMALLER is a partial removal, so charge the
    # difference: that is what catches "the manifest says 2 nodes, the target
    # holds 50,000".
    total_nodes = count("MATCH (n:Entity) RETURN count(n)", {})
    doomed = sum(live_counts.get(t, 0) for t in removed)
    doomed += sum(
        max(0, live_counts.get(t, 0) - int(manifest_repos.get(t, {}).get("node_count") or 0))
        for t in changed
    )
    if not allow_shrink and total_nodes > 0 and (doomed / total_nodes) > shrink_limit:
        raise ValueError(
            f"graphify: delta push would remove {doomed} of "
            f"{total_nodes} nodes ({doomed / total_nodes:.0%}) in the target "
            f"graph, over the {shrink_limit:.0%} safety limit. That usually "
            f"means this manifest does not belong to this database — check "
            f"--graph-name. Pass --allow-shrink if it is intended. Nothing was "
            f"changed."
        )

    nodes_pushed = edges_pushed = deleted = 0
    for tag in changed:
        deleted += _prune_repo_paged(run, count, tag)
        for chunk in _chunked(_repo_nodes(G, tag), _BATCH):
            stamped = []
            for nid, attrs in chunk:
                cid = node_community.get(nid)
                if cid is not None:
                    attrs["community"] = cid
                attrs[_EPOCH_PROP] = epoch
                stamped.append((nid, attrs))
            target.add_nodes_from(stamped)
            nodes_pushed += len(stamped)
        for chunk in _chunked(_repo_edges(G, tag), _BATCH):
            stamped = [(u, v, {**a, _EPOCH_PROP: epoch}) for u, v, a in chunk]
            target.add_edges_from(stamped)
            edges_pushed += len(stamped)
        info = manifest_repos.get(tag, {})
        run(
            f"MERGE (s:{_STATE_LABEL} {{repo: $repo}}) "
            f"SET s.source_hash = $h, s.node_count = $n, s.epoch = $e",
            {"repo": tag, "h": info.get("source_hash"),
             "n": int(info.get("node_count") or 0), "e": epoch},
        )

    for tag in removed:
        deleted += _prune_repo_paged(run, count, tag)
        run(f"MATCH (s:{_STATE_LABEL} {{repo: $repo}}) DELETE s", {"repo": tag})

    return {
        "nodes": nodes_pushed,
        "edges": edges_pushed,
        "deleted": deleted,
        "deleted_edges": 0,  # repo prune is DETACH DELETE; edges go with the nodes
        "repos_pushed": changed,
        "repos_removed": removed,
        "repos_skipped": [t for t in manifest_repos if t not in changed],
        "reasons": reasons,
    }


def push_to_falkordb(
    G,
    uri: str,
    user: str | None = None,
    password: str | None = None,
    communities: dict[int, list[str]] | None = None,
    graph_name: str = "graphify",
    prune: bool = False,
    allow_shrink: bool = False,
    shrink_limit: float = _DEFAULT_SHRINK_LIMIT,
    repo_manifest: dict | None = None,
) -> dict[str, int]:
    """Push graph directly to a running FalkorDB instance.

    Requires: pip install falkordb

    Writes through ``GraphStore``, the same batched ``UNWIND`` writer graphify
    uses for its own graphs, so the pushed graph gets the ``:Entity`` label, the
    ``n.id`` index, and a schema ``graphify query``/``serve`` can read back.
    Only the host/port are read from the URI, so the scheme is informational —
    "falkordb://localhost:6379", "redis://localhost:6379" and a bare
    "localhost:6379" are all equivalent (default port 6379). Auth is optional
    (FalkorDB runs without credentials by default), so user and password may be
    None; credentials embedded in the URI take precedence.

    graph_name: which named graph in the instance to write (FalkorDB keys each
        graph by name, so this is the difference between a staging graph and
        production — always set it explicitly for anything that matters).
    prune: delete nodes and edges this push did not write, so the target
        converges on the source instead of accumulating. See the module
        docstring.
    repo_manifest: the global manifest's ``repos`` dict. Switches the push into
        repo-keyed DELTA mode — only repos whose ``source_hash`` moved (or whose
        node count in the target has drifted from the manifest) are re-sent, and
        repos the manifest no longer lists are deleted. Convergence is implied,
        so ``prune`` is not needed with it.

    Returns a dict with counts of nodes and edges pushed, nodes and edges
    deleted, and in delta mode the repos re-pushed / skipped / removed.
    """
    from graphify.store import GraphStore

    node_community = _node_community_map(communities) if communities else {}
    epoch = _new_epoch()

    # GraphStore.__init__ creates the :Entity(id) index and applies the same
    # optional-auth rules as the old inline connection code.
    target = GraphStore(graph_name=graph_name, uri=uri, user=user, password=password)

    # See the Neo4j path: label pre-schema nodes so MERGE matches them instead
    # of creating a duplicate beside each one.
    target._g.query("MATCH (n) WHERE n.id IS NOT NULL AND NOT n:Entity SET n:Entity")

    def run(cypher, params):
        target._g.query(cypher, params)

    def count(cypher, params):
        return int(target._g.query(cypher, params).result_set[0][0])

    def rows(cypher, params):
        return target._g.query(cypher, params).result_set or []

    if repo_manifest is not None:
        return _push_delta(
            G, target, node_community, epoch, repo_manifest,
            run, count, rows, allow_shrink, shrink_limit,
        )

    nodes_pushed = 0
    edges_pushed = 0
    for chunk in _chunked(_stamped_nodes(G, node_community, epoch), _BATCH):
        target.add_nodes_from(chunk)
        nodes_pushed += len(chunk)
    for chunk in _chunked(_stamped_edges(G, epoch), _BATCH):
        target.add_edges_from(chunk)
        edges_pushed += len(chunk)

    deleted = deleted_edges = 0
    if prune:
        deleted, deleted_edges = _converge(run, count, epoch, allow_shrink, shrink_limit)

    return {
        "nodes": nodes_pushed,
        "edges": edges_pushed,
        "deleted": deleted,
        "deleted_edges": deleted_edges,
    }
