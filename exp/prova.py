import dgl
from dgl.data import BACommunityDataset


def create_custom_dataset(inter_edges):
    dataset = BACommunityDataset(num_inter_edges=inter_edges)
    dataset.process()

    #print(dataset.num_classes)
    g = dataset[0]
    g = dgl.to_networkx(g, node_attrs = ["feat", "label"])
    g = from_networkx(g)

    # Definitions of node features and labels 
    g.x = g.feat.float()
    g.y = g.label

    n_training = g.num_nodes // 100 * 60
    n_test = g.num_nodes // 100 * 20
    #print(n_test * 3)
    split_list = torch.zeros(g.num_nodes)
    split_list[n_training:n_training+n_test] = 1
    split_list[n_training+n_test:] = 2
    #print(sum(split_list))
    #print(split_list[:30])
    perm = torch.randperm(g.num_nodes)
    split_list = split_list[perm[:]]
    #print(sum(split_list))
    #print(split_list[100:200])

    mask_train = split_list == 0
    mask_valid = split_list == 1
    mask_test = split_list == 2
    #mask_train = mask[perm[:]]
    #mask_test = ~mask_train

    # Training data
    g.train_mask = mask_train
    g.test_mask = mask_test
    g.val_mask = mask_valid

    return g

dataset = create_custom_dataset(350)
print(dataset)