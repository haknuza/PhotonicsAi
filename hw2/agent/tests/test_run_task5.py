import run_task5


def test_default_mat_path_finds_task6_data_when_local_data_is_missing(tmp_path, monkeypatch):
    repo = tmp_path
    agent_dir = repo / "hw2" / "agent"
    task6_data = repo / "Task6" / "silverbox-agents" / "data" / "SNLS80mV.mat"
    task6_data.parent.mkdir(parents=True)
    task6_data.touch()
    monkeypatch.setattr(run_task5, "HERE", agent_dir)

    assert run_task5.default_mat_path() == task6_data.resolve()
