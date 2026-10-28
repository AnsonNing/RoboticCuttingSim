import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def checker():
    path=Path(__file__).resolve().parents[1]/"scripts/run_isaac_sim_cutting.py"
    spec=importlib.util.spec_from_file_location("native_source_recorder",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_frozen_sources


@pytest.mark.parametrize("changed_file",["source","snapshot"])
def test_shared_startup_source_or_snapshot_race_is_preserved_and_blocks(tmp_path,changed_file):
    root=tmp_path/"root";root.mkdir()
    output=tmp_path/"episode";(output/"implementation").mkdir(parents=True)
    source=root/"rig.py";snapshot=output/"implementation/rig.py"
    source.write_text("mount = 1\n");snapshot.write_bytes(source.read_bytes())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    (output/"manifest.json").write_text(json.dumps({"implementation_sha256":{"rig.py":digest}}))
    verify=checker()
    verify(output,"before_app",root)
    (source if changed_file=="source" else snapshot).write_text("mount = 2\n")
    with pytest.raises(RuntimeError,match="Frozen native source changed"):
        verify(output,"before_live_reset",root)
    assert json.loads((output/"source_integrity_before_app.json").read_text())["status"]=="PASS"
    failure=json.loads((output/"source_integrity_before_live_reset.json").read_text())
    assert failure["status"]=="FAIL" and failure["mismatches"][0]["frozen_sha256"]==digest
