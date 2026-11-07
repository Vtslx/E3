# d1 run failed

`RuntimeError: shape '[1, 4, 1, 128]' is invalid for input of size 2048`

```
Traceback (most recent call last):
  File "/Volumes/Net/E3/src/sprm/cli.py", line 458, in _cmd_verify
    outcome = gate_d1(
        backend,
    ...<5 lines>...
        max_new_tokens=overrides.get("max_new_tokens"),
    )
  File "/Volumes/Net/E3/src/sprm/eval/gates.py", line 1532, in gate_d1
    runs = run_experiment(
        backend,
    ...<5 lines>...
        archive=archive,
    )
  File "/Volumes/Net/E3/src/sprm/eval/runner.py", line 502, in run_experiment
    run = run_sample(
        backend,
    ...<3 lines>...
        max_new_tokens=max_new_tokens,
    )
  File "/Volumes/Net/E3/src/sprm/eval/runner.py", line 454, in run_sample
    inject_query_read(backend, online.cache, online.build, question_queries, experiment)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Volumes/Net/E3/src/sprm/runtime/form_a.py", line 640, in inject_query_read
    key_row = key.reshape(1, info.num_kv_heads, 1, info.head_dim).to(layer.keys.dtype)
              ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[1, 4, 1, 128]' is invalid for input of size 2048

```