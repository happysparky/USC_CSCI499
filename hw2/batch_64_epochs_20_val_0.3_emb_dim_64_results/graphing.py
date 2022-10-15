import matplotlib.pyplot as plt

# note, must be in the same directly as the metric files to run this!! 
def graph_and_save_loss(f_name):
    with open(f_name, "r") as file:
        loss = []
        for line in file:
            line = line.strip()
            line = line.split(" ")
            loss.append(line[0])

    x = [i for i in range(0, len(loss))]
    plt.clf()
    plt.plot(x, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f_name[:f_name.index("_")]+" loss")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f_name[:f_name.index("_")]+"_loss")

graph_and_save_loss("train_metrics.txt")
graph_and_save_loss("val_metrics.txt")