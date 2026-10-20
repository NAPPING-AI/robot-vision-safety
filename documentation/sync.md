
## Command to sync local folder to project on the backpack

```bash
rsync -avz --exclude='.git/' unitree@192.168.123.164:~/unitree_sdk2_python/ ./
```