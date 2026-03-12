from tensorboard.backend.event_processing import event_accumulator

# Đường dẫn tới file event
event_file = "events.out.tfevents.1765986614.cea12525cbc9.70 (1).0"

# Load event
ea = event_accumulator.EventAccumulator(
    event_file,
    size_guidance={
        event_accumulator.SCALARS: 0,  # load toàn bộ scalar
    }
)
ea.Reload()

# Xem tất cả scalar có trong log
print("Available scalars:")
print(ea.Tags()['scalars'])
# Lấy scalar mAP
map_events = ea.Scalars('train_loss')

for e in map_events:
    print(f"Step={e.step}, Value={e.value:.4f}")