# Raw measurements (Raspberry Pi 5)

Benchmark output behind Tables 4 and 7 of the paper, copied verbatim from the
board. Nothing here was recomputed on the desktop.

## Files

| File | What it holds |
|---|---|
| `pi_latency_powersave.json` | Model-inference latency of every exported ONNX model: 10 warm-up runs followed by 100 timed runs, reported as mean / worst / minimum and FPS. |
| `pi_power_powersave.json` | Board power, energy per image, peak RSS and die temperature for the same models, read from the PMIC rails (`vcgencmd pmic_read_adc`). |

## Session

Raspberry Pi 5 (4 GB), Debian 12, kernel 6.12.96, ONNX Runtime 1.29.0 with
`CPUExecutionProvider`, governor `powersave`, four intra-op threads, graph
optimization `ORT_ENABLE_ALL`, random input of the stated shape.

## Mapping to the paper

| Paper label | File in the archive |
|---|---|
| A (640x640) | `A_640_seed42_640.onnx` |
| B (1280x1280) | `baseline_seed42_1280.onnx` |
| D (1280x1280 + BiFlow) | `full_seed42_1280.onnx` |

Table 4 of the paper reports the latency file. Table 7 reports power, energy
and peak memory from the power file.

## Note on the two sessions

Latency and power were recorded in two separate runs on the same board. The
power session reports its own mean latency (A 202.12 ms, B 938.89 ms,
D 930.40 ms), which differs slightly from the latency benchmark (A 192.38 ms,
B 894.49 ms, D 955.66 ms). Energy per image is the product of inference power
and the mean latency of the same session, so the power and energy columns stay
internally consistent. The paper states both sets of values.
