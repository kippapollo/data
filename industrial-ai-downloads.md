# Industrial AI software and model downloads

This catalog covers the 108 projects in the [CubeStudio industrial AI architecture](industrial-ai-open-source-architecture.html). The model tables identify what you can download and what the plant team supplies.

## Model training recipes

The platform fits these models on plant data. The first motor pilot uses the fault-classification workflow. These links provide implementations and training tools. The architecture publishes no pretrained motor-fault weights.

| Task and model | Download | What to fit or build |
| --- | --- | --- |
| Forecast a sensor value · ARIMA | [statsmodels ARIMA source](https://raw.githubusercontent.com/statsmodels/statsmodels/7a4c633d664e287da07f561600ddb3ca700dafb5/statsmodels/tsa/arima/model.py) · [Package files](https://pypi.org/project/statsmodels/#files) | Fit one timestamped series and compare against a last-value or seasonal baseline. |
| Classify a known fault · Random Forest | [scikit-learn forest source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1f98d2b4a5ddea0c251201f555cbeeeb8756165e/sklearn/ensemble/_forest.py) · [CubeStudio training launcher](https://raw.githubusercontent.com/data-infra/cube-studio/4506db899ec12317aa62deb92fbe0de3b38838ee/job-template/job/ray_sklearn/launcher.py) | Use RandomForestClassifier with labeled sensor features. Keep windows from each machine run in the same split. |
| Flag unusual operation · Isolation Forest | [scikit-learn Isolation Forest source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1f98d2b4a5ddea0c251201f555cbeeeb8756165e/sklearn/ensemble/_iforest.py) · [Package files](https://pypi.org/project/scikit-learn/#files) | Fit normal operating periods and choose the alert threshold on separate validation data. |
| Classify vibration windows · custom 1D CNN | [PyTorch Conv1d layer source](https://raw.githubusercontent.com/pytorch/pytorch/3f915f45bf5af116f418578034e2a8442e72425e/torch/nn/modules/conv.py) · [CPU wheels](https://download.pytorch.org/whl/cpu/torch/) | Implement the network and train it with labeled vibration windows. Conv1d supplies the convolution layer. |

The optional LLM path names vLLM and orchestration libraries. It leaves the model, checkpoint and tokenizer to the deployment owner. The proposed English portal, site admission service and plant adapters also require implementation.

## Datasets for CubeStudio

Start with CubeStudio’s two example CSVs to check the training and prediction workflow. For a first industrial task, the steel-energy CSV needs only two columns for ARIMA. The bearing recordings are a closer match for the motor pilot and require a preprocessing step.

| Dataset | Recipe | Prepare the training and evaluation data | Downloads and source |
| --- | --- | --- | --- |
| CubeStudio scikit-learn example | Random Forest workflow check | Set `y` as the target and use the other ten numeric columns as inputs. The unlabeled file repeats examples from the labeled file and is useful for checking prediction output. Use a separate holdout to measure accuracy. | [Labeled CSV](https://raw.githubusercontent.com/data-infra/cube-studio/4506db899ec12317aa62deb92fbe0de3b38838ee/myapp/example/pipeline/sklearn/ml-label.csv) · [Prediction CSV](https://raw.githubusercontent.com/data-infra/cube-studio/4506db899ec12317aa62deb92fbe0de3b38838ee/myapp/example/pipeline/sklearn/ml-nolabel.csv) |
| Steel Industry Energy Consumption | ARIMA electricity-use forecast | Unzip `Steel_industry_data.csv`. Parse `date` day-first, sort by time, and forecast `Usage_kWh` at 15-minute intervals. Reserve the latest dates for testing. | [CSV in ZIP](https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip) · [UCI dataset and license](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption) |
| AI4I 2020 Predictive Maintenance | Random Forest failure classification | Use the five numeric process measurements to predict `Machine failure` in 10,000 synthetic records. Exclude the two IDs and all five failure-mode flags from the inputs because the flags disclose the outcome. | [CSV in ZIP](https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip) · [UCI dataset and license](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| Skoltech Anomaly Benchmark (SKAB) | Isolation Forest on pump sensors | Read the semicolon-delimited CSVs. Fit on sensor columns from the normal run. Score a separate valve experiment and compare predictions with its `anomaly` column. Keep `anomaly`, `changepoint` and the timestamp out of the feature matrix. | [Normal-run CSV](https://raw.githubusercontent.com/waico/SKAB/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data/anomaly-free/anomaly-free.csv) · [Labeled valve-run CSV](https://raw.githubusercontent.com/waico/SKAB/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data/valve1/0.csv) · [Experiment guide](https://github.com/waico/SKAB/tree/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data) |
| Case Western Reserve University bearing vibration | Custom 1D CNN for motor faults | The two sample MAT files contain 12 kHz drive-end recordings at zero motor load. Extract the channel ending in `_DE_time` and label it by defect type. For evaluation, download additional runs and assign whole recordings to splits before cutting them into windows. | [Inner-race fault MAT](https://engineering.case.edu/sites/default/files/105.mat) · [Ball fault MAT](https://engineering.case.edu/sites/default/files/118.mat) · [All 12 kHz drive-end runs](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) |
| Condition Monitoring of Hydraulic Systems | Random Forest component-condition classification | Each row in a sensor file represents one 60-second test cycle. Compute per-cycle features and join them by row to `profile.txt`. For valve condition, use the second profile column as the target. | [Sensor matrices and labels (ZIP)](https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip) · [UCI dataset and license](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| Biomass properties and gasification results | Custom tabular regression for syngas composition | Predict `volH2 (%)` from feedstock properties and operating inputs such as temperature and steam-to-biomass ratio. Exclude all gas composition and yield outputs from the inputs. Hold out entire biomass types to test transfer to a new feedstock. | [Gasification CSV](https://zenodo.org/records/7358117/files/data_GASIF_biomass.csv) · [Authors’ dataset record](https://zenodo.org/records/7358117) |

## Example models for sector simulations

These upstream examples can be opened with tools in the architecture. A deployment supplies its own equipment parameters and site-specific cases.

| Tool | Downloadable example | Contents |
| --- | --- | --- |
| pandapower | [IEEE 14-bus network (JSON)](https://raw.githubusercontent.com/e2nIEE/pandapower/912026744dcc0ae4bc2bc3c05ac0c20ace211c29/pandapower/networks/power_system_test_case_jsons/case14.json) | Power-flow test network. |
| PyPSA | [Meshed AC/DC network (NetCDF)](https://raw.githubusercontent.com/PyPSA/PyPSA/a042f1a7e4a80ad3701742010cce82d9615594eb/examples/networks/ac-dc-meshed/ac-dc-meshed.nc) | Example electrical network for optimization studies. |
| GridLAB-D | [IEEE 13-node feeder (GLM)](https://raw.githubusercontent.com/gridlab-d/gridlab-d/e1841e1eebced819e45209f4a899763ce337b177/models/IEEE_13_Node_Test_Feeder.glm) | Distribution feeder model. |
| IDAES | [Process flowsheet with costing (notebook)](https://raw.githubusercontent.com/IDAES/examples/main/idaes_examples/notebooks/docs/flowsheets/hda_flowsheet_with_costing.ipynb) | Hydrodealkylation flowsheet with an economic model. |
| DWSIM | [Cooling tower flowsheet (DWXMZ)](https://raw.githubusercontent.com/DanWBR/dwsim/1abf72d1b6b41d3e9a8cc770d3cc4e8fc76e5766/PlatformFiles/Common/FOSSEE/Cooling_Tower/CoolingTower.dwxmz) | Process simulation from the bundled FOSSEE examples. |

## Software downloads

Use the package, installer or release link to choose a build for your platform. Source ZIPs point to fixed upstream commits. They use the architecture’s recorded revision where available, including CubeStudio at `4506db899ec1`. Other ZIPs capture an upstream source revision checked on 24 November 2026. The architecture documents the build and qualification steps separately. **Bold project names** identify software selected for the CPU platform, including components that run only on the regional cluster.

### Applications and analysis

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| Superset | Read-only analytical dashboards | [Package files](https://pypi.org/project/apache-superset/#files) | [Source ZIP](https://github.com/apache/superset/archive/de1162834c347200779f4dfce39201f98f37be41.zip) |
| Metabase | Self-service queries and dashboards (OSS edition) | [OSS JAR](https://downloads.metabase.com/latest/metabase.jar) | [Source ZIP](https://github.com/metabase/metabase/archive/b5d0223e3011c6815aa58de0709d561a5e500574.zip) |
| **Grafana** | Read-only plant and service dashboards | [OSS builds](https://grafana.com/grafana/download?edition=oss) | [Source ZIP](https://github.com/grafana/grafana/archive/af93b89f0eab2ad6d1fd76b05ce0ee7980fe4b92.zip) |

### AI engineering and model lifecycle

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **CubeStudio** | Community training workflows and staged inference | [Installer at reviewed revision](https://github.com/data-infra/cube-studio/blob/4506db899ec12317aa62deb92fbe0de3b38838ee/install/kubernetes/start.sh) | [Source ZIP](https://github.com/data-infra/cube-studio/archive/4506db899ec12317aa62deb92fbe0de3b38838ee.zip) |
| **MLflow** | Experiment tracking, evaluation artifacts and model registry | [Package files](https://pypi.org/project/mlflow/#files) | [Source ZIP](https://github.com/mlflow/mlflow/archive/f9fc58bbd532c4b81a063c55ff6d16bc0f0b54bc.zip) |
| DVC | Version datasets and model artifacts | [Package files](https://pypi.org/project/dvc/#files) | [Source ZIP](https://github.com/treeverse/dvc/archive/b59add99ee667f2b352cc1be337619f965fe3fea.zip) |
| JupyterLab | Interactive engineering notebooks | [Package files](https://pypi.org/project/jupyterlab/#files) | [Source ZIP](https://github.com/jupyterlab/jupyterlab/archive/fb21ae758c4bfe6f361d6b2d99b3a25b03532a28.zip) |
| Optuna | Hyperparameter search | [Package files](https://pypi.org/project/optuna/#files) | [Source ZIP](https://github.com/optuna/optuna/archive/c44f2ce3cdc8f096f83897e0089fbf7e1375f70c.zip) |
| Ray | Distributed training and compute | [Package files](https://pypi.org/project/ray/#files) | [Source ZIP](https://github.com/ray-project/ray/archive/456e2863a1f8d4d2d74cd1a886636b83f6b200bd.zip) |

### Prediction and optimization

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **statsmodels** | ARIMA forecasting | [Package files](https://pypi.org/project/statsmodels/#files) | [Source ZIP](https://github.com/statsmodels/statsmodels/archive/7a4c633d664e287da07f561600ddb3ca700dafb5.zip) |
| **PyTorch** | Train and run the custom 1D CNN on CPU | [CPU wheels](https://download.pytorch.org/whl/cpu/torch/) | [Source ZIP](https://github.com/pytorch/pytorch/archive/3f915f45bf5af116f418578034e2a8442e72425e.zip) |
| **scikit-learn** | Preprocessing, fault classifiers and anomaly detection | [Package files](https://pypi.org/project/scikit-learn/#files) | [Source ZIP](https://github.com/scikit-learn/scikit-learn/archive/1f98d2b4a5ddea0c251201f555cbeeeb8756165e.zip) |
| XGBoost | Gradient-boosted trees for tabular prediction | [Package files](https://pypi.org/project/xgboost/#files) | [Source ZIP](https://github.com/dmlc/xgboost/archive/aae18e572531e102c974292102692fdd8a0e6dc2.zip) |
| LightGBM | Gradient boosting for large tabular datasets | [Package files](https://pypi.org/project/lightgbm/#files) | [Source ZIP](https://github.com/lightgbm-org/LightGBM/archive/53350cd896c6fd76d2cc03354f452f18646830e9.zip) |
| CatBoost | Boosted trees for categorical and numeric features | [Package files](https://pypi.org/project/catboost/#files) | [Source ZIP](https://github.com/catboost/catboost/archive/cc0ce40bee61db190a9d1ebc67d442ad5d89862c.zip) |
| HiGHS | Linear, quadratic and mixed-integer optimization | [Package files](https://pypi.org/project/highspy/#files) | [Source ZIP](https://github.com/ERGO-Code/HiGHS/archive/f3f113d230429a8c0014b12ab0dd1a3847e58c55.zip) |
| OR-Tools | Scheduling, routing and constraint solving | [Package files](https://pypi.org/project/ortools/#files) | [Source ZIP](https://github.com/google/or-tools/archive/100f66e6242ab8bf8d32feb8f3bf086db66ae2b5.zip) |
| Pyomo | Formulate optimization problems for a solver | [Package files](https://pypi.org/project/pyomo/#files) | [Source ZIP](https://github.com/Pyomo/pyomo/archive/4c7b1cf789b1ba064e42374fcebaba9087ca8f98.zip) |
| CVXPY | Formulate convex optimization problems | [Package files](https://pypi.org/project/cvxpy/#files) | [Source ZIP](https://github.com/cvxpy/cvxpy/archive/0fc467a64a66f5a0f3b7cc721779dca5f1b7ce6f.zip) |

### LLM and agent composition

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| vLLM | Optional inference engine for separately selected LLM weights | [Package files](https://pypi.org/project/vllm/#files) | [Source ZIP](https://github.com/vllm-project/vllm/archive/c84273be384d7faaeadfb36caced2f9d5a112f7f.zip) |
| Haystack | Compose retrieval and LLM pipelines | [Package files](https://pypi.org/project/haystack-ai/#files) | [Source ZIP](https://github.com/deepset-ai/haystack/archive/3b488d9fdcad3f59c06553cb87c01d86994f4cac.zip) |
| LlamaIndex | Index documents and retrieve model context | [Package files](https://pypi.org/project/llama-index/#files) | [Source ZIP](https://github.com/run-llama/llama_index/archive/99d7433f6f71888951379ccca9294e272442c832.zip) |
| LangGraph | Stateful agent workflows | [Package files](https://pypi.org/project/langgraph/#files) | [Source ZIP](https://github.com/langchain-ai/langgraph/archive/02f2c2e6c992ffccce9e9129781fdd04f015c6fa.zip) |
| LangChain | Compose LLM applications | [Package files](https://pypi.org/project/langchain/#files) | [Source ZIP](https://github.com/langchain-ai/langchain/archive/92c7defb0ac54ca2be18fca1bcf89cc88d9356c6.zip) |

### Model representation and serving

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **FastAPI** | Backend for the proposed English portal and inference APIs | [Package files](https://pypi.org/project/fastapi/#files) | [Source ZIP](https://github.com/fastapi/fastapi/archive/05e6df7e906c3decba377c37548019973f40bf15.zip) |
| **ONNX** | Portable model graph format and tools | [Package files](https://pypi.org/project/onnx/#files) | [Source ZIP](https://github.com/onnx/onnx/archive/366620ca097ed2dff8f76b50f264dd592d9edc9b.zip) |
| **ONNX Runtime** | CPU inference for compatible exported models | [Package files](https://pypi.org/project/onnxruntime/#files) | Snapshot unavailable at review. [Upstream install guide](https://onnxruntime.ai/docs/install/) |
| KServe | Cloud model-serving controller | [Releases](https://github.com/kserve/kserve/releases) | [Source ZIP](https://github.com/kserve/kserve/archive/b93083bd1d4388b88d9b94f09bbe64bd257e3f60.zip) |
| BentoML | Package a model as an inference API | [Package files](https://pypi.org/project/bentoml/#files) | [Source ZIP](https://github.com/bentoml/BentoML/archive/adb91c7c8169b00ee04c1f0a804afebe0c845cfd.zip) |
| Triton Inference Server | Serve multiple models with build-specific backends | [Releases](https://github.com/triton-inference-server/server/releases) | [Source ZIP](https://github.com/triton-inference-server/server/archive/24af411c99d09cfb60259dceb63a718606148aa2.zip) |

### Twin and simulation

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| Eclipse Ditto | Manage digital-twin state and connectivity | [Releases](https://github.com/eclipse-ditto/ditto/releases) | [Source ZIP](https://github.com/eclipse-ditto/ditto/archive/8aeabbe8a2d10804c44a2dcb93059d9193f2b314.zip) |
| BaSyx | Asset Administration Shell services (Java server SDK) | [Releases](https://github.com/eclipse-basyx/basyx-java-server-sdk/releases) | [Source ZIP](https://github.com/eclipse-basyx/basyx-java-server-sdk/archive/5417b0cd6b63bc53694e4f6730499cfe3b0e5c5f.zip) |
| FMPy | Run and inspect Functional Mock-up Units | [Package files](https://pypi.org/project/FMPy/#files) | [Source ZIP](https://github.com/CATIA-Systems/FMPy/archive/e42e288f448830c5f146ae334b38e91d17ff19fa.zip) |
| OpenModelica | Compile and simulate Modelica models | [Installers](https://openmodelica.org/download/) | [Source ZIP](https://github.com/OpenModelica/OpenModelica/archive/60072b7794bde5442487da90c669762e210ad42e.zip) |
| OpenFOAM | Computational fluid dynamics (Foundation distribution) | [Foundation downloads](https://openfoam.org/download/) | [Source ZIP](https://github.com/OpenFOAM/OpenFOAM-dev/archive/d78e9ac4ef06daf72f5350a5c3dce3f9794688d9.zip) |
| SU2 | Flow simulation and engineering optimization | [Releases](https://github.com/su2code/SU2/releases) | [Source ZIP](https://github.com/su2code/SU2/archive/6c0e5c5c720936b175c4004e57332884443437f5.zip) |
| SimPy | Discrete-event simulation of queues and material flow | [Package files](https://pypi.org/project/simpy/#files) | [Source tarball](https://files.pythonhosted.org/packages/4a/38/fb393d1a5c2f35b65609e8d3c39d1508ada6a767c7e624ffa16ed3e84ffd/simpy-4.1.2.tar.gz) |
| pandapower | Electrical network models and power-flow analysis | [Package files](https://pypi.org/project/pandapower/#files) | [Source ZIP](https://github.com/e2nIEE/pandapower/archive/912026744dcc0ae4bc2bc3c05ac0c20ace211c29.zip) |
| PyPSA | Power-system capacity and dispatch optimization | [Package files](https://pypi.org/project/pypsa/#files) | [Source ZIP](https://github.com/PyPSA/PyPSA/archive/a042f1a7e4a80ad3701742010cce82d9615594eb.zip) |
| GridLAB-D | Electric distribution-grid simulation | [Releases](https://github.com/gridlab-d/gridlab-d/releases) | [Source ZIP](https://github.com/gridlab-d/gridlab-d/archive/e1841e1eebced819e45209f4a899763ce337b177.zip) |
| IDAES | Process models, flowsheets and optimization | [Package files](https://pypi.org/project/idaes-pse/#files) | [Source ZIP](https://github.com/IDAES/idaes-pse/archive/d8b68063b9f380824855a6bff662d12261e37378.zip) |
| DWSIM | Chemical process flowsheet simulation | [Releases](https://github.com/DanWBR/dwsim/releases) | [Source ZIP](https://github.com/DanWBR/dwsim/archive/1abf72d1b6b41d3e9a8cc770d3cc4e8fc76e5766.zip) |

### Data engineering and governance

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| Kafka | Durable event streams and replay | [Distributions](https://kafka.apache.org/downloads/) | [Source ZIP](https://github.com/apache/kafka/archive/c3cbf35565aaf8754f266320f8f8a73a7189abd2.zip) |
| Debezium | Capture database changes as events | [Connectors](https://debezium.io/releases/) | [Source ZIP](https://github.com/debezium/debezium/archive/8e97518e6b1d6dac19e58c0537e53bc776fd0c14.zip) |
| NiFi | Ingest, route and transform data | [Distributions](https://nifi.apache.org/download/) | [Source ZIP](https://github.com/apache/nifi/archive/a8473d374c976660e4ef3aa932936adaa1e21dbd.zip) |
| Flink | Process streams using event time and windows | [Distributions](https://flink.apache.org/downloads/) | [Source ZIP](https://github.com/apache/flink/archive/d8301306ef69825109ec264dc2cd1b8ae8575059.zip) |
| dbt | SQL transformations (dbt source framework) | [Core Python package](https://pypi.org/project/dbt-core/#files) | [Source ZIP](https://github.com/dbt-labs/dbt/archive/5ec87d85a6a0a2726595b6c6c66fb32079584d23.zip) |
| Great Expectations | Validate dataset expectations | [Package files](https://pypi.org/project/great-expectations/#files) | [Source ZIP](https://github.com/fivetran/great_expectations/archive/e08d1a800c8b759c84b820e420f5699392c53a98.zip) |
| Airflow | Schedule and monitor batch workflows | [Package files](https://pypi.org/project/apache-airflow/#files) | [Source ZIP](https://github.com/apache/airflow/archive/f5f7f2b838b95dc4b76d2ecb1069429a3c4af1c3.zip) |
| OpenMetadata | Catalog data assets, ownership and lineage | [Releases](https://github.com/open-metadata/OpenMetadata/releases) | [Source ZIP](https://github.com/open-metadata/OpenMetadata/archive/c34d08956adc92508dd6455192a455322a0169bc.zip) |

### Data stores and queries

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **Valkey** | Redis-compatible task broker and cache for CubeStudio | [Releases](https://valkey.io/download/) | [Source ZIP](https://github.com/valkey-io/valkey/archive/5b16df2e55e3ab5e530085fd6fec312bf55f1b99.zip) |
| **MySQL Community Server** | CubeStudio metadata database | [Community Server builds](https://dev.mysql.com/downloads/mysql/) | [Source ZIP](https://github.com/mysql/mysql-server/archive/bf5a74e48423ac51200987edcd72a678772cb2ee.zip) |
| **PostgreSQL** | Domain records, receipts and MLflow metadata | [Packages](https://www.postgresql.org/download/) | [Source ZIP](https://github.com/postgres/postgres/archive/11591f273e6a26fb55728b410538fa2627b19b28.zip) |
| **Ceph** | Shared workspaces (CephFS) and S3-compatible objects (RGW) | [Packages](https://download.ceph.com/) | [Source ZIP](https://github.com/ceph/ceph/archive/274f6de963d5b422625f579d5f99da532c294247.zip) |
| Iceberg | Versioned analytical table format | [Releases](https://iceberg.apache.org/releases/) | [Source ZIP](https://github.com/apache/iceberg/archive/2517ec2a7f70721166bd2b2ca0061984e4eb2bdc.zip) |
| Trino | Distributed SQL over approved data sources | [Server and CLI](https://trino.io/download.html) | [Source ZIP](https://github.com/trinodb/trino/archive/d54888f41fbd8b34a51cfad7dce8c9b31b0512b9.zip) |
| **DuckDB** | Embedded SQL for offline analysis | [Package files](https://pypi.org/project/duckdb/#files) | [Source ZIP](https://github.com/duckdb/duckdb/archive/629e395a854d72dcb1f414e1a0451c68ac0e9f1f.zip) |
| OpenSearch | Optional document and log indexes | [Distributions](https://opensearch.org/downloads.html) | [Source ZIP](https://github.com/opensearch-project/OpenSearch/archive/00d9641bebbb21ca83158738967307ef96ad1d4a.zip) |

### Industrial connectivity

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **open62541** | Read-only OPC UA acquisition adapter | [Releases](https://github.com/open62541/open62541/releases) | [Source ZIP](https://github.com/open62541/open62541/archive/59f556e9920d76e8ecc6567b890003b54561d447.zip) |
| NodeOPCUA | OPC UA client/server stack for Node.js | [npm tarball](https://registry.npmjs.org/node-opcua/-/node-opcua-2.187.34.tgz) | [Source ZIP](https://github.com/node-opcua/node-opcua/archive/86c0da2bc582a06e6a8f8531905b6da1ea376c8f.zip) |
| PLC4X | Industrial protocol drivers | [Distributions](https://plc4x.apache.org/plc4x/latest/users/download.html) | [Source ZIP](https://github.com/apache/plc4x/archive/3ef3cdd75505b26338f757a3321097fc37696069.zip) |
| Paho | Eclipse Paho C MQTT client for device adapters | [Releases](https://github.com/eclipse-paho/paho.mqtt.c/releases) | [Source ZIP](https://github.com/eclipse-paho/paho.mqtt.c/archive/4a939ddb01eea581a32fd6f0adcfee51b91d2601.zip) |
| **Mosquitto** | Local MQTT broker for observations | [Packages](https://mosquitto.org/download/) | [Source ZIP](https://github.com/eclipse-mosquitto/mosquitto/archive/7ae66ca5cf5e75dca18ba2f6cbdd21453c8e51a5.zip) |
| Hono | Tenant-aware device messaging | [Releases](https://github.com/eclipse-hono/hono/releases) | [Source ZIP](https://github.com/eclipse-hono/hono/archive/e1c98bec09efad17203a5ca8b470763c7fb11294.zip) |
| EdgeX | Device services and edge data acquisition (edgex-go) | [Releases](https://github.com/edgexfoundry/edgex-go/releases) | [Source ZIP](https://github.com/edgexfoundry/edgex-go/archive/c0d70ab6588425cef004cd03aa4178659f882796.zip) |
| Zenoh | Edge publish/subscribe, storage and queries | [Releases](https://github.com/eclipse-zenoh/zenoh/releases) | [Source ZIP](https://github.com/eclipse-zenoh/zenoh/archive/72ea967c9fe8f19d770e6e7f99fa293132754d19.zip) |
| ChirpStack | LoRaWAN network server for sensor telemetry | [Releases](https://github.com/chirpstack/chirpstack/releases) | [Source ZIP](https://github.com/chirpstack/chirpstack/archive/3f8dad84da772b06b14fb8975e9d20318baac525.zip) |
| StreamPipes | Industrial sensor-data pipelines | [Distributions](https://streampipes.apache.org/download/) | [Source ZIP](https://github.com/apache/streampipes/archive/feeb42f5b42c293c4ddfdbc861009a55632f6758.zip) |
| Node-RED | Prototype read-only integration flows | [npm tarball](https://registry.npmjs.org/node-red/-/node-red-5.0.10.tgz) | [Source ZIP](https://github.com/node-red/node-red/archive/f324616ae3e38cd1d21b8ff7637adaac1e314b4c.zip) |

### Workflow and software delivery

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| Temporal | Run durable business workflows (server) | [Releases](https://github.com/temporalio/temporal/releases) | [Source ZIP](https://github.com/temporalio/temporal/archive/4197dfdd52eb338c932e64db209e3853fc7a032d.zip) |
| **Argo CD** | Reconcile Kubernetes deployments from Git | [Releases](https://github.com/argoproj/argo-cd/releases) | [Source ZIP](https://github.com/argoproj/argo-cd/archive/973bf65cc1d142cbdc105f8c91ddc372fdc3829e.zip) |
| Flux | Alternative GitOps controllers and CLI | [Releases](https://github.com/fluxcd/flux2/releases) | [Source ZIP](https://github.com/fluxcd/flux2/archive/d7e87c943d611c0521e9be87ff3ceb7081b936a8.zip) |
| **Helm** | Package Kubernetes applications | [Releases](https://github.com/helm/helm/releases) | [Source ZIP](https://github.com/helm/helm/archive/3c2635de452c8df481f59a1e1246bef1355b7e8d.zip) |
| Tekton | Build and test pipelines on Kubernetes | [Releases](https://github.com/tektoncd/pipeline/releases) | [Source ZIP](https://github.com/tektoncd/pipeline/archive/a71b0a65d86ec7116fc70c770cb7df71ef63a47f.zip) |
| **Argo Workflows** | Orchestrate CubeStudio training jobs | [Releases](https://github.com/argoproj/argo-workflows/releases) | [Source ZIP](https://github.com/argoproj/argo-workflows/archive/0fbdf6fb3b55f8f0d55f293faafb82461769380f.zip) |
| **Harbor** | Registry for signed container images and release artifacts | [Releases](https://github.com/goharbor/harbor/releases) | [Source ZIP](https://github.com/goharbor/harbor/archive/e7dbdea223486b177ce87b6c954eaad756dd0b4f.zip) |

### Observability and inventory

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **OpenTelemetry** | Collect and export telemetry (core Collector) | [Collector builds](https://github.com/open-telemetry/opentelemetry-collector-releases/releases) | [Source ZIP](https://github.com/open-telemetry/opentelemetry-collector/archive/1fcf8d3233e04fae3bee7639e035c7c69d5da2a3.zip) |
| Jaeger | Distributed tracing | [Binaries](https://www.jaegertracing.io/download/) | [Source ZIP](https://github.com/jaegertracing/jaeger/archive/8e07bbf4ba7b19adb11e2c10dc1ab46c2f2344fc.zip) |
| **Prometheus** | Service metrics and alerting | [Binaries](https://prometheus.io/download/) | [Source ZIP](https://github.com/prometheus/prometheus/archive/a5022ef622b4718c3dda1291c1e641fbed44c607.zip) |
| Thanos | Federate and retain Prometheus metrics | [Releases](https://github.com/thanos-io/thanos/releases) | [Source ZIP](https://github.com/thanos-io/thanos/archive/9d58b7a93550ea9514005be822615a07da6736c5.zip) |
| Loki | Optional aggregated logs | [Releases](https://github.com/grafana/loki/releases) | [Source ZIP](https://github.com/grafana/loki/archive/c1ab23aede2bbb78f92fde4effc01eb0288a6691.zip) |
| NetBox | Infrastructure inventory | [Releases](https://github.com/netbox-community/netbox/releases) | [Source ZIP](https://github.com/netbox-community/netbox/archive/112a6282ea2484bf520982a56b08f68af21974f1.zip) |

### Cloud and runtime

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **Debian** | CPU host operating system (Debian main) | [Installation images](https://www.debian.org/distrib/) | [Main source archive](https://deb.debian.org/debian/pool/main/) |
| **Volcano** | Schedule batch training jobs | [Releases](https://github.com/volcano-sh/volcano/releases) | [Source ZIP](https://github.com/volcano-sh/volcano/archive/2006640ed881e632119eedb9d3fac82819402301.zip) |
| **Kubernetes** | Orchestrate regional workloads | [Components](https://kubernetes.io/releases/download/) | [Source ZIP](https://github.com/kubernetes/kubernetes/archive/f9951a0479e395b4d13155dbe4f15611993e3ac0.zip) |
| KubeEdge | Optional edge workload management | [Releases](https://github.com/kubeedge/kubeedge/releases) | [Source ZIP](https://github.com/kubeedge/kubeedge/archive/3a9cb19fba9b799cf9062acaf656b62a4acad92b.zip) |
| OpenStack | Private-cloud compute (Nova source) | [Nova source releases](https://releases.openstack.org/teams/nova.html) | [Source ZIP](https://github.com/openstack/nova/archive/e2f5b0300b937f74571cc82c8962dc8a64986503.zip) |
| Proxmox VE | Virtualization host (pve-manager source) | [VE ISO images](https://www.proxmox.com/en/downloads/proxmox-virtual-environment/iso) | [Source ZIP](https://github.com/proxmox/pve-manager/archive/7382f81a79979a14a84421a9053e989a2c68e751.zip) |
| OpenTofu | Infrastructure provisioning | [Releases](https://github.com/opentofu/opentofu/releases) | [Source ZIP](https://github.com/opentofu/opentofu/archive/fd9945e839f9acc7604045a1a227934d1783db93.zip) |
| Ansible Core | Host configuration and automation | [Package files](https://pypi.org/project/ansible-core/#files) | [Source ZIP](https://github.com/ansible/ansible/archive/b4b2b1e82a535d27f902a71fc8396c3af138fca2.zip) |
| Velero | Back up and restore Kubernetes resources | [Releases](https://github.com/velero-io/velero/releases) | [Source ZIP](https://github.com/velero-io/velero/archive/290a05a756f021e649105257c664970ce2f99805.zip) |
| **containerd** | Cluster container runtime | [Releases](https://github.com/containerd/containerd/releases) | [Source ZIP](https://github.com/containerd/containerd/archive/4ba4c8933013e4532b0d678bedf82f00bacd1476.zip) |
| **Podman** | Run approved model-serving containers at the plant | [Releases](https://github.com/podman-container-tools/podman/releases) | [Source ZIP](https://github.com/podman-container-tools/podman/archive/0eab672d838be47083cbd1398f748ac74da8cf44.zip) |

### Network and service boundaries

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **Istio** | Route and secure inference traffic through Envoy | [Releases](https://github.com/istio/istio/releases) | [Source ZIP](https://github.com/istio/istio/archive/b1490474a639bce48fb9bc74982916d60b15179c.zip) |
| Cilium | Alternative workload networking and policy | [Releases](https://github.com/cilium/cilium/releases) | [Source ZIP](https://github.com/cilium/cilium/archive/719594740515ce71c1f13082225f6eb87d6528da.zip) |
| **Calico** | Cluster networking and network policy | [Releases](https://github.com/projectcalico/calico/releases) | [Source ZIP](https://github.com/projectcalico/calico/archive/741a0a6b39b1205a1285c37215b67812d3497fd7.zip) |
| **Envoy** | Service and API proxy | [Releases](https://github.com/envoyproxy/envoy/releases) | [Source ZIP](https://github.com/envoyproxy/envoy/archive/0d7446a18fa5fff164f9a60baf4dc53f0dae6d6b.zip) |
| APISIX | API gateway for approved services | [Distributions](https://apisix.apache.org/downloads/) | [Source ZIP](https://github.com/apache/apisix/archive/bb7a303af5019031a675c79e766da2884a51237c.zip) |
| Traefik | Ingress and reverse proxy (Traefik Proxy) | [Releases](https://github.com/traefik/traefik/releases) | [Source ZIP](https://github.com/traefik/traefik/archive/8be77b01c1b246400ee080aefb148b640d765a10.zip) |

### Cross-layer security

| Project | Use in the architecture | Download or install | Source |
| --- | --- | --- | --- |
| **Keycloak** | Project login and identity federation | [Server builds](https://www.keycloak.org/downloads) | [Source ZIP](https://github.com/keycloak/keycloak/archive/7d055b70e2a4e9309937a66f1de7d1a0a288d693.zip) |
| SPIRE | Issue and attest workload identities | [Releases](https://github.com/spiffe/spire/releases) | [Source ZIP](https://github.com/spiffe/spire/archive/e1d196adafd35d6aca8f29d13592d1cdcc5c8ecc.zip) |
| **OPA** | Evaluate scoped access policies | [Releases](https://github.com/open-policy-agent/opa/releases) | [Source ZIP](https://github.com/open-policy-agent/opa/archive/7b037a97dd9b465dedb1c565f778ea06ac5569ea.zip) |
| **OpenBao** | Manage credentials, secrets and PKI | [Releases](https://github.com/openbao/openbao/releases) | [Source ZIP](https://github.com/openbao/openbao/archive/a238d70d33f18716daecf08ce6b52f789e7d20f4.zip) |
| **cert-manager** | Manage Kubernetes certificate lifecycles | [Releases](https://github.com/cert-manager/cert-manager/releases) | [Source ZIP](https://github.com/cert-manager/cert-manager/archive/192296167e04942debef61f4cb52e97a5066aebc.zip) |
| **Cosign** | Sign and verify release artifacts | [Releases](https://github.com/sigstore/cosign/releases) | [Source ZIP](https://github.com/sigstore/cosign/archive/19c90f91263701e6e3fb9f6e32ae1a8199712baf.zip) |
| **Syft** | Generate software bills of materials | [Releases](https://github.com/anchore/syft/releases) | [Source ZIP](https://github.com/anchore/syft/archive/9ad9aa5e80a6762c33b0344d252ec669ae53afdc.zip) |
| **Trivy** | Scan container images and artifacts | [Releases](https://github.com/aquasecurity/trivy/releases) | [Source ZIP](https://github.com/aquasecurity/trivy/archive/a7b0d649ea07bf957acdb7a742edc81d27358f2c.zip) |
| Falco | Detect unexpected runtime behavior | [Releases](https://github.com/falcosecurity/falco/releases) | [Source ZIP](https://github.com/falcosecurity/falco/archive/74ccf76b7b2d85ff42c97da6bd6a716b0d1d17b9.zip) |

## Supporting dependencies

| Component | Use in the architecture | Download |
| --- | --- | --- |
| Git | Version control for cluster configuration and delivery. | [Downloads](https://git-scm.com/downloads) |
| Python | Runtime for native ARIMA and other Python model bundles. | [Releases](https://www.python.org/downloads/) |
| Ceph CSI | Mount CephFS workspaces in Kubernetes. | [Releases](https://github.com/ceph/ceph-csi/releases) |
| Ant Design | UI components used by the upstream CubeStudio frontend. | [npm tarball](https://registry.npmjs.org/antd/-/antd-6.7.2.tgz) |

The reference stack uses Valkey for the upstream Redis 7.4 broker role. For the responsibilities of the selected components, see the [architecture’s stack table](industrial-ai-open-source-architecture.html#selected-stack).
