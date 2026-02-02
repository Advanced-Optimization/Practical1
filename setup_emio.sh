#!/usr/bin/bash
mv ~/emio-labs/v25.12.01/assets/labs/labsConfig.json ~/emio-labs/v25.12.01/assets/labs/labsConfig.json.bkp
ln -fs "$(realpath labsConfig.json)" ~/emio-labs/v25.12.01/assets/labs/
ln -fs "$(realpath .)" ~/emio-labs/v25.12.01/assets/labs/
