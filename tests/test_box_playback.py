"""Terminal telemetry must describe the finished episode, not the reset state."""

from isaaclab_tutorial.tasks.place_box.inspect_policy import merge_terminal_state


def test_terminal_samples_replace_reset_values_only_for_finished_environments():
    current = {"box_position": [[0, 0, 0.08], [1, 1, 0.08]], "lifted": [False, False]}
    terminal = {0: {"box_position": [0.2, 0.1, 0.08], "lifted": True}}
    result = merge_terminal_state(current, terminal)
    assert result["box_position"] == [[0.2, 0.1, 0.08], [1, 1, 0.08]]
    assert result["lifted"] == [True, False]
