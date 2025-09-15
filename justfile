gaugenetwork:
    uv run scripts/gauge_network.py data/river_network/river_network.gpkg data/gauge_q_loc.csv data/gauge_network.json

lateralflow:
    uv run scripts/lateralflow_create.py data/qlat data/wrf_data.nc data/river_network/river_weight.arrow

passthrough:
    #!/usr/bin/env sh
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_01.parquet data/qout_passthrough/qout_01.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_02.parquet data/qout_passthrough/qout_02.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_03.parquet data/qout_passthrough/qout_03.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_04.parquet data/qout_passthrough/qout_04.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_05.parquet data/qout_passthrough/qout_05.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_06.parquet data/qout_passthrough/qout_06.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_07.parquet data/qout_passthrough/qout_07.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_08.parquet data/qout_passthrough/qout_08.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_09.parquet data/qout_passthrough/qout_09.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_10.parquet data/qout_passthrough/qout_10.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_11.parquet data/qout_passthrough/qout_11.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_12.parquet data/qout_passthrough/qout_12.parquet --model=pass_through &
    uv run scripts/routing.py data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_13.parquet data/qout_passthrough/qout_13.parquet --model=pass_through &
    wait


calibration:
    #!/usr/bin/env sh
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_01.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_01.parquet data/qout_calibration/qout_01_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_02.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_02.parquet data/qout_calibration/qout_02_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_03.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_03.parquet data/qout_calibration/qout_03_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_04.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_04.parquet data/qout_calibration/qout_04_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_05.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_05.parquet data/qout_calibration/qout_05_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_06.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_06.parquet data/qout_calibration/qout_06_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_07.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_07.parquet data/qout_calibration/qout_07_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_08.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_08.parquet data/qout_calibration/qout_08_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_09.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_09.parquet data/qout_calibration/qout_09_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_10.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_10.parquet data/qout_calibration/qout_10_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_11.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_11.parquet data/qout_calibration/qout_11_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_12.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_12.parquet data/qout_calibration/qout_12_parameter.json &
    uv run scripts/calibration.py --model muskingum data/river_network/river_setup.arrow data/river_network/river_parameter.arrow 20130501T000000.arrow 300 data/qlat/qlat_13.parquet data/gauge_network.json data/gauge_q_obs.parquet data/qout_calibration/qout_13.parquet data/qout_calibration/qout_13_parameter.json &
    wait


ensemble:
    uv run scripts/routing_ensemble_generate.py data/river_network/river_parameter.arrow data/gauge_network.json data/qout_calibration/celerity_measurement.csv 100 data/qout_ensemble/

routing_ensemble:
    uv run scripts/routing_ensemble.py data/river_network/river_setup.arrow data/qout_ensemble/ data/20130501T000000.arrow 300 data/qlat data/qout_ensemble data/gauge_network.json --max_proc 12

plot_domain:
    uv run scripts/plot_domain.py

plot_calibration:
    uv run scripts/plot_calibration.py

plot_streamflow:
    uv run scripts/plot_streamflow.py

plot_griddeddata:
    uv run scripts/plot_griddeddata.py

plot_relationship:
    uv run scripts/plot_relationship.py
