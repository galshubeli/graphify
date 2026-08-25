"""Integration test for push_to_falkordb against a real FalkorDB instance.

Runs for real against `falkordb/falkordb:latest`:

    docker run -d -p 6379:6379 falkordb/falkordb:latest
    uv run pytest tests/test_falkordb_integration.py -q

The test auto-skips when the `falkordb` SDK is not installed or no FalkorDB is
reachable, so it is a no-op in the default CI (which runs no external services).
Host/port are overridable via FALKORDB_HOST / FALKORDB_PORT.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

falkordb = pytest.importorskip("falkordb")

FIXTURES = Path(__file__).parent / "fixtures"
HOST = os.environ.get("FALKORDB_HOST", "localhost")
PORT = int(os.environ.get("FALKORDB_PORT", "6379"))
GRAPH_NAME = "graphify_test"


def _connect():
    """Return a connected FalkorDB client, or skip if none is reachable."""
    try:
        db = falkordb.FalkorDB(host=HOST, port=PORT)
        db.connection.ping()
        return db
    except Exception as e:  # pragma: no cover - depends on local environment
        pytest.skip(f"no FalkorDB reachable at {HOST}:{PORT} ({e})")


@pytest.fixture()
def db():
    client = _connect()
    # Start from a clean slate and clean up afterwards.
    try:
        client.select_graph(GRAPH_NAME).delete()
    except Exception:
        pass
    yield client
    try:
        client.select_graph(GRAPH_NAME).delete()
    except Exception:
        pass


def test_push_to_falkordb_creates_expected_graph(db):
    from graphify.build import build_from_json
    from graphify.export import push_to_falkordb

    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    G = build_from_json(extraction)

    result = push_to_falkordb(
        G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME
    )

    assert result["nodes"] == G.number_of_nodes()
    assert result["edges"] == G.number_of_edges()

    graph = db.select_graph(GRAPH_NAME)
    node_count = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    edge_count = graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]

    assert node_count == G.number_of_nodes()
    assert edge_count == G.number_of_edges()


def test_push_to_falkordb_is_idempotent(db):
    """MERGE-based push is safe to re-run - counts must not grow."""
    from graphify.build import build_from_json
    from graphify.export import push_to_falkordb

    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    G = build_from_json(extraction)

    push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)
    push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)

    graph = db.select_graph(GRAPH_NAME)
    node_count = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    edge_count = graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]

    assert node_count == G.number_of_nodes()
    assert edge_count == G.number_of_edges()


def _counts(db, name=GRAPH_NAME):
    graph = db.select_graph(name)
    n = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    e = graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
    return n, e


def _build(extraction_overrides=None):
    from graphify.build import build_from_json

    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    if extraction_overrides:
        extraction = extraction_overrides(extraction)
    return build_from_json(extraction, graph_name="graphify_push_src", uri=f"{HOST}:{PORT}")


def test_pushed_nodes_carry_the_entity_label(db):
    """A pushed graph must be readable by graphify's own store, which matches
    on :Entity. Before #3057 the exporter labelled by file_type only, so every
    `graphify query`/`serve` against a pushed graph came back empty."""
    from graphify.export import push_to_falkordb

    G = _build()
    push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)

    graph = db.select_graph(GRAPH_NAME)
    total = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    entities = graph.query("MATCH (n:Entity) RETURN count(n)").result_set[0][0]
    assert entities == total > 0


def test_push_without_prune_still_never_deletes(db):
    """The default stays add-only, so the old contract is unchanged."""
    from graphify.export import push_to_falkordb

    G = _build()
    push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)
    db.select_graph(GRAPH_NAME).query(
        "CREATE (:Entity {id: 'stale-node-1'})"
    )
    before, _ = _counts(db)

    result = push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)

    after, _ = _counts(db)
    assert after == before
    assert result["deleted"] == 0


def test_prune_removes_what_the_source_no_longer_has(db):
    """#3057: with --prune a re-push converges on the source instead of
    accumulating the pruned nodes forever."""
    from graphify.export import push_to_falkordb

    G = _build()
    push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)
    ground_truth = _counts(db)

    # Simulate what `global add` leaves behind: nodes the source has dropped.
    graph = db.select_graph(GRAPH_NAME)
    for i in range(5):
        graph.query(f"CREATE (:Entity {{id: 'pruned-{i}'}})")
    assert _counts(db)[0] == ground_truth[0] + 5

    result = push_to_falkordb(
        G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, prune=True, allow_shrink=True
    )

    assert result["deleted"] == 5
    # Identical to a from-scratch load on both counts.
    assert _counts(db) == ground_truth
    survivors = graph.query(
        "MATCH (n:Entity) WHERE n.id STARTS WITH 'pruned-' RETURN count(n)"
    ).result_set[0][0]
    assert survivors == 0


def test_prune_is_idempotent_and_a_noop_when_nothing_is_stale(db):
    from graphify.export import push_to_falkordb

    G = _build()
    push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, prune=True)
    baseline = _counts(db)

    result = push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, prune=True)

    assert result["deleted"] == 0
    assert _counts(db) == baseline


def test_prune_refuses_a_mass_deletion_without_allow_shrink(db):
    """The #479 rule applied to the push: a manifest aimed at the wrong graph
    looks exactly like a genuine mass removal, so refuse it by default."""
    from graphify.export import push_to_falkordb

    G = _build()
    graph = db.select_graph(GRAPH_NAME)
    # A target dominated by nodes this push does not write.
    for i in range(200):
        graph.query(f"CREATE (:Entity {{id: 'someone-elses-{i}'}})")
    before, _ = _counts(db)

    with pytest.raises(ValueError, match="safety limit"):
        push_to_falkordb(
            G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, prune=True
        )

    # Refused means refused: nothing deleted.
    survivors = graph.query(
        "MATCH (n:Entity) WHERE n.id STARTS WITH 'someone-elses-' RETURN count(n)"
    ).result_set[0][0]
    assert survivors == 200
    assert _counts(db)[0] >= before


def test_graph_name_isolates_targets(db):
    """Before #3057 the CLI could not name a target, so every push landed on
    the `graphify` key. Two names must not touch each other."""
    from graphify.export import push_to_falkordb

    other = f"{GRAPH_NAME}_other"
    try:
        G = _build()
        push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)
        push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=other)
        db.select_graph(other).query("CREATE (:Entity {id: 'only-in-other'})")

        push_to_falkordb(
            G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, prune=True, allow_shrink=True
        )

        still_there = db.select_graph(other).query(
            "MATCH (n:Entity {id: 'only-in-other'}) RETURN count(n)"
        ).result_set[0][0]
        assert still_there == 1
    finally:
        try:
            db.select_graph(other).delete()
        except Exception:
            pass


def test_prune_removes_an_edge_the_source_dropped(db):
    """Convergence has to cover edges, not just nodes. DETACH DELETE takes the
    edges of a pruned node with it, but an edge dropped from the source whose
    two endpoints both survive needs its own sweep — that is the surplus-edge
    half of #3057 (the report measured +1,151 edges alongside +1,250 nodes)."""
    from graphify.export import push_to_falkordb
    from graphify.store import GraphStore

    src = GraphStore(graph_name="graphify_push_src", uri=f"{HOST}:{PORT}")
    src.clear()
    src.add_nodes_from([(f"n{i}", {"label": f"s{i}", "file_type": "python"}) for i in range(10)])
    src.add_edges_from([(f"n{i}", f"n{i + 1}", {"relation": "calls"}) for i in range(9)])

    push_to_falkordb(src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, prune=True)
    assert _counts(db) == (10, 9)

    src.remove_edges([("n0", "n1")])  # both endpoints survive

    result = push_to_falkordb(
        src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, prune=True, allow_shrink=True
    )

    assert result["deleted"] == 0        # no node went away
    assert result["deleted_edges"] == 1  # but the dropped edge did
    assert _counts(db) == (10, 8)


# --------------------------------------------------------------------------
# Repo-keyed delta (#3057)
# --------------------------------------------------------------------------

SRC_GLOBAL = "graphify_global_src"


@pytest.fixture()
def global_src():
    """A two-repo global graph shaped the way `global add` leaves one."""
    from graphify.store import GraphStore

    G = GraphStore(graph_name=SRC_GLOBAL, uri=f"{HOST}:{PORT}")
    G.clear()
    G.add_nodes_from(
        [(f"repoA::n{i}", {"label": f"a{i}", "file_type": "python", "repo": "repoA"})
         for i in range(20)]
        + [(f"repoB::n{i}", {"label": f"b{i}", "file_type": "python", "repo": "repoB"})
           for i in range(20)]
    )
    G.add_edges_from(
        [(f"repoA::n{i}", f"repoA::n{i + 1}", {"relation": "calls"}) for i in range(19)]
        + [(f"repoB::n{i}", f"repoB::n{i + 1}", {"relation": "calls"}) for i in range(19)]
        # cross-repo edge: B depends on a node owned by A
        + [("repoB::n0", "repoA::n0", {"relation": "imports"})]
    )
    yield G
    try:
        G.clear()
    except Exception:
        pass


def _entity_counts(db, name=GRAPH_NAME):
    """Nodes/edges excluding the :GraphifyPushState bookkeeping nodes."""
    graph = db.select_graph(name)
    n = graph.query("MATCH (n:Entity) RETURN count(n)").result_set[0][0]
    e = graph.query("MATCH (:Entity)-[r]->(:Entity) RETURN count(r)").result_set[0][0]
    return n, e


def _manifest(G, tags=("repoA", "repoB"), hashes=None):
    hashes = hashes or {}
    out = {}
    for t in tags:
        n = int(G._g.query(
            "MATCH (n:Entity {repo:$t}) RETURN count(n)", {"t": t}
        ).result_set[0][0])
        out[t] = {"source_hash": hashes.get(t, f"hash-{t}-v1"), "node_count": n}
    return out


def test_delta_skips_repos_whose_hash_has_not_moved(global_src):
    """The delta mirrors global_add's own contract: an unmoved source_hash is a
    skip, so a 226-repo graph with one changed repo sends one repo's rows."""
    from graphify.export import push_to_falkordb

    m = _manifest(global_src)
    first = push_to_falkordb(
        global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m
    )
    assert sorted(first["repos_pushed"]) == ["repoA", "repoB"]

    second = push_to_falkordb(
        global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m
    )
    assert second["repos_pushed"] == []
    assert sorted(second["repos_skipped"]) == ["repoA", "repoB"]
    assert second["nodes"] == 0 and second["edges"] == 0


def test_delta_resends_only_the_changed_repo(db, global_src):
    from graphify.export import push_to_falkordb

    m = _manifest(global_src)
    push_to_falkordb(global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m)
    before = _entity_counts(db)

    m["repoA"]["source_hash"] = "hash-repoA-v2"  # repoA changed, repoB did not
    result = push_to_falkordb(
        global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m
    )

    assert result["repos_pushed"] == ["repoA"]
    assert result["repos_skipped"] == ["repoB"]
    assert result["nodes"] == 20            # repoA only, not all 40
    assert _entity_counts(db) == before            # and the graph is unchanged in shape


def test_delta_preserves_the_cross_repo_edge_when_a_repo_is_rewritten(db, global_src):
    """Pruning repoA drops the B->A edge with A's node. Re-adding only edges
    whose BOTH endpoints are in repoA would lose it silently — the delta has to
    restore every edge incident to the repo."""
    from graphify.export import push_to_falkordb

    m = _manifest(global_src)
    push_to_falkordb(global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m)
    graph = db.select_graph(GRAPH_NAME)
    q = ("MATCH (:Entity {id:'repoB::n0'})-[r:IMPORTS]->(:Entity {id:'repoA::n0'}) "
         "RETURN count(r)")
    assert graph.query(q).result_set[0][0] == 1

    m["repoA"]["source_hash"] = "hash-repoA-v2"
    push_to_falkordb(global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m)

    assert graph.query(q).result_set[0][0] == 1, "cross-repo edge lost by the delta"


def test_delta_deletes_a_repo_the_manifest_no_longer_lists(db, global_src):
    """#3057's core defect, at the repo level: what the source pruned must go."""
    from graphify.export import push_to_falkordb

    m = _manifest(global_src)
    push_to_falkordb(global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m)
    assert _entity_counts(db)[0] == 40

    del m["repoB"]
    result = push_to_falkordb(
        global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME,
        repo_manifest=m, allow_shrink=True,
    )

    assert result["repos_removed"] == ["repoB"]
    assert result["deleted"] == 20
    graph = db.select_graph(GRAPH_NAME)
    assert graph.query(
        "MATCH (n:Entity {repo:'repoB'}) RETURN count(n)"
    ).result_set[0][0] == 0


def test_delta_repairs_drift_a_ledger_would_miss(db, global_src):
    """A 'what I last pushed' ledger still reads clean after the database is
    wiped or a run half-lands. Reading the target's OWN per-repo counts turns
    that silent permanent drift into automatic repair."""
    from graphify.export import push_to_falkordb

    m = _manifest(global_src)
    push_to_falkordb(global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m)

    # Half-landed run: someone/something removed part of repoB from the target.
    db.select_graph(GRAPH_NAME).query(
        "MATCH (n:Entity {repo:'repoB'}) WITH n LIMIT 8 DETACH DELETE n"
    )
    assert _entity_counts(db)[0] == 32

    # Hashes have NOT moved — a ledger-only delta would skip both repos here.
    result = push_to_falkordb(
        global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m
    )

    assert result["repos_pushed"] == ["repoB"]
    assert "drift" in result["reasons"]["repoB"]
    assert _entity_counts(db)[0] == 40  # repaired


def test_delta_refuses_a_manifest_aimed_at_the_wrong_database(db, global_src):
    """The 1,211,189-node accident: a 40-repo test manifest pointed at a live
    graph looks exactly like a genuine mass removal."""
    from graphify.export import push_to_falkordb

    m = _manifest(global_src)
    push_to_falkordb(global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=m)

    foreign = {"someone-elses-repo": {"source_hash": "x", "node_count": 1}}
    with pytest.raises(ValueError, match="safety limit"):
        push_to_falkordb(
            global_src, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, repo_manifest=foreign
        )

    assert _entity_counts(db)[0] == 40  # nothing touched


def test_add_only_push_reports_the_drift_it_cannot_fix(db):
    """#3057's divergence is silent. An add-only push can't converge, but it
    can say how far the target has drifted (@Azeem1985's review request)."""
    from graphify.export import push_to_falkordb

    G = _build()
    push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)
    db.select_graph(GRAPH_NAME).query("CREATE (:Entity {id: 'left-behind'})")

    result = push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)

    assert result["deleted"] == 0          # still add-only
    assert result["target_surplus"] == 1   # but no longer silent about it

    # A converged target reports no surplus.
    converged = push_to_falkordb(
        G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME, prune=True, allow_shrink=True
    )
    assert converged["deleted"] == 1
    after = push_to_falkordb(G, uri=f"{HOST}:{PORT}", graph_name=GRAPH_NAME)
    assert after["target_surplus"] == 0
