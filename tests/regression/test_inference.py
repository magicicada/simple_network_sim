import pandas as pd

from simple_network_sim import inference as inf


def test_inference(base_data_dir):
    summary = inf.run_inference(str(base_data_dir / "config_inference.yaml"))

    assert summary["best_distance"] == 21.23091271317998
    assert summary["weights"] == [4.535230195917359e-12, 1.2577973135105333e-14]
    assert summary["distances"] == [21.232393437779567, 21.23091271317998]
    pd.testing.assert_frame_equal(summary["best_particle"].inferred_variables["infection-probability"].value,
                                  pd.DataFrame([{"Date": "2020-03-09", "Value": 0.148430698780}]))
    pd.testing.assert_frame_equal(summary["best_particle"].inferred_variables["initial-infections"].value,
                                  pd.DataFrame([{"Health_Board": "S08000016", "Age": "[17,70)", "Infected": 79.77746}]))
    pd.testing.assert_frame_equal(
        summary["best_particle"].inferred_variables["contact-multipliers"].value,
        pd.DataFrame([
          {"Date": "2020-03-16", "Movement_Multiplier": 1., "Contact_Multiplier": 1.128966},
          {"Date": "2020-05-05", "Movement_Multiplier": .05, "Contact_Multiplier": 0.210406},
          {"Date": "2020-05-30", "Movement_Multiplier": .3, "Contact_Multiplier": 0.210406},
          {"Date": "2020-06-04", "Movement_Multiplier": .8, "Contact_Multiplier": 0.210406},
          {"Date": "2020-06-24", "Movement_Multiplier": .9, "Contact_Multiplier": 0.210406},
        ])
    )
