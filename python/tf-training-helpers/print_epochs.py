import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--input", type=str, default="epochs.log")
args = parser.parse_args()

log = open(args.input).read().splitlines()

learning_rate = 0.001
monitor = "val_loss"

early_stopping = 20
reduce_patience = 15
reduce_factor = 0.20000008989515007

header = log[0].split(";")

if len(header) == 5:
	header = [header[0]] + header[1:3][::-1] + header[3:5][::-1]

monitor_index = header.index(monitor)

best_value = 9999
best_line = 0
patience = 0

for i, line in enumerate(log[1:]):
	line = [float(x) for x in line.split(";")]
	line[0] = i + 1

	if len(line) == 5:
		line = [line[0]] + line[1:3][::-1] + line[3:5][::-1]	
	
	l1 = f"xxx/xxx [============================>.] - ETA: 0s"
	
	for y in range(1, len(header)):
		l1 += f" - {header[y]}: {line[y]:.4f}"
	
	if line[monitor_index] < best_value:
		patience = 0
		l2 = f"Epoch {line[0]}: {monitor} improved from {best_value:.5f} to {line[monitor_index]:.5f}, saving model to ../output/<DATASET>/<METHOD>/checkpoint_weights.hdf5"
		best_value = line[monitor_index]
		best_line = line
	else:
		patience += 1
		l2 = f"Epoch {line[0]}: {monitor} did not improve from {best_value:.5f}"
	
	l2_5 = ""
	
	if patience == reduce_patience:
		patience = 0
		learning_rate = learning_rate * reduce_factor
		l2_5 = f"\n\nEpoch {line[0]}: ReduceLROnPlateau reducing learning rate to {learning_rate}."
	
	l3 = f"xxx/xxx [==============================] - xxs 272ms/step"
	
	for y in range(1, len(header)):
		l3 += f" - {header[y]}: {line[y]:.4f}"
	
	print(f"Epoch {line[0]}/1000\n{l1}\n{l2}{l2_5}\n{l3}")


fin = "\nEarlyStopping." if len(log[1:]) - (best_line[0]) == early_stopping else "\nNo EarlyStopping."
fin += f"\nFinal Learning Rate: {learning_rate}"
fin += f"\nLast Best Epoch: {int(best_line[0])}"

for y in range(1, len(header)):
	fin += f" - {header[y]}: {best_line[y]:.4f}"
	
print(f"{fin}\n")

