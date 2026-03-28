# 监控docker资源消耗
```bash
sudo python3 docker_monitory.py $DOCKER_NAME -i 1 -m 3600 -w 60 -f i2v_20251203_fp8.log -c i2v_20251203_fp8.csv
```

# 可视化结果，判断是否内存泄漏

```bash
sudo python3 show_stat.py i2v_20251203_fp8.csv
```

# 测试device-device, host->device， device->host速度
```bash
python multi_device_communication_test.py
```