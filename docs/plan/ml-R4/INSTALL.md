# Installation

The archive contains a top-level `ml-R4/` directory.

From the SENTINEL repository root:

```bash
rm -rf /tmp/sentinel-ml-r4-package
mkdir -p /tmp/sentinel-ml-r4-package
unzip /path/to/sentinel_ml_R4_label_recovery_package.zip -d /tmp/sentinel-ml-r4-package

mkdir -p docs/plan
rm -rf docs/plan/ml-R4
cp -a /tmp/sentinel-ml-r4-package/ml-R4 docs/plan/ml-R4

find docs/plan/ml-R4 -maxdepth 2 -type f | sort
```

Do not run `rm -rf docs/plan/ml-R4` if it contains local results that have not been backed up. In that case, copy the existing directory first and merge deliberately.

After installation, give the agent the instruction contained in:

`docs/plan/ml-R4/START_HERE_AGENT.md`
