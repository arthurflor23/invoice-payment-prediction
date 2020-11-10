import datetime

epochs = open("epochs.log", "r").read().splitlines()
train = open("train.txt", "r").read().splitlines()

epochs2 = open("epochs2.log", "w")
train2 = open("train2.txt", "w")

total_epochs = len(epochs) - 1

for i in range(len(epochs)):
    if i > 0:
        splitted = epochs[i].split(";")
        splitted[0] = str(i - 1)
        epochs[i] = ";".join(splitted)

    epochs2.write(epochs[i] + "\n")
    
for i in range(len(train)):
    if str.startswith(train[i], "Total epochs"):
        train[i] = train[i].replace(train[i].split()[-1], str(total_epochs))

    if str.startswith(train[i], "Total time"):
        time = train[i + 1].split()[-1].split(":")

        time = [int(y) for x in time for y in x.split(".")]
        time_per_epoch = datetime.timedelta(hours=time[0], minutes=time[1], seconds=time[2], microseconds=time[3])

        train[i] = train[i].replace(train[i].split()[-1], f"{(total_epochs * time_per_epoch)}")

    if str.startswith(train[i], "Best epoch"):
        train[i] = train[i].replace(train[i].split()[-1], str(total_epochs - 20))

    train2.write(train[i] + "\n")

epochs2.close()
train2.close()