import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from cvxopt import matrix, solvers
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

class logsticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(logsticRegression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        Theta_x = self.logstic(x)
        out = F.softmax(self.logstic(x), dim=1)
        return out, Theta_x

def robust_fair_loss(Theta_X, weight, Sen, Sen_bar):

    weight, Sen = weight.view(len(weight)), Sen.view(len(Sen))
    Theta_X = torch.sum(Theta_X, dim=1)
    Theta_X = Theta_X.view(len(Theta_X))

    fair_loss = torch.mul(torch.mul(weight, Sen) - Sen_bar, Theta_X)

    return torch.mean(torch.mul(fair_loss, fair_loss))

# Can be put on GPU, since it uses PyTorch!! yay
def compute_model_parameter_fairness(weight, Sen, ratio, model, train_loader, optimizer, num_epochs=10):
    Sen = torch.from_numpy(Sen).float()

    for _ in range(num_epochs):

        running_acc = 0.0

        model.train()

        criterion = nn.CrossEntropyLoss(reduction='none')

        for i,  (data, target) in enumerate(train_loader):
            img, label = data, target

            label = label.data.numpy()

            for i in range(len(label)):
                label[i] = 0 if label[i] == 0.0 else 1

            label = torch.LongTensor(label)

            label = label.view(len(label))

            out, Theta_x = model(img)

            loss = criterion(out, label)

            loss_vector = loss

            loss = loss.view(len(loss), 1)
            weight = weight.view(len(weight), 1)
            loss = torch.mul(loss, weight)

            loss = loss.sum() / len(out)

            _, pred = torch.max(out.data, 1)

            loss = loss + robust_fair_loss(Theta_x, weight, Sen, ratio)
            
            running_acc += torch.sum((pred==label))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        running_acc = running_acc.numpy()

    return loss_vector.detach().numpy(), Theta_x.detach().numpy()

# Turn into GPU friendly version for Kmeans 
def preprocess(X, Y, S, cluster_size=300, random_state=0):

    kmeans = KMeans(n_clusters=cluster_size, random_state=random_state).fit(X)

    cluster_list = np.array(kmeans.labels_)

    maps = {}

    for class_label in cluster_list:
        maps[class_label] = maps.get(class_label, 0) + 1

    cluster = []
    real_upper = []
    real_lower = []
    rho = 0.02
    index = 0
    weight = [0.0]*len(cluster_list)

    for key in sorted(maps.keys()):
        value = maps[key]

        for i in range(index, index + value):
            weight[i] = value / len(weight)

        lambda_hat = value / len(weight)

        if lambda_hat - rho < 0.01:
            real_lower.append(0.02)
        else:
            real_lower.append(lambda_hat - rho)

        real_upper.append(lambda_hat + rho)
        cluster.append(value)
        index += value

    upper_bound = []
    lower_bound = []

    for i in range(len(real_lower)):
        upper_bound.append(1.8)
        lower_bound.append(-0.2)

    I = np.argsort(cluster_list.flatten())
    S = S[I,None].astype(float)
    X = X[I]
    Y = Y[I]

    return X, Y, S, lower_bound, upper_bound, cluster

class UnlabeledFairRobust():
    def __init__(self):
        pass

    def fit(self, X, Y, S, n_iters=20, learning_rate=0.25, batch_size=None, num_epochs=10):

        X, Y, S, lower, upper, cluster = preprocess(X, Y, S)
        batch_size = len(X) if batch_size is None else batch_size
        self.model = logsticRegression(X.shape[1], 2)
        optimizer  = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        weight     = torch.from_numpy(np.ones(len(X))).float()
        ratio_fair = np.sum(S) / len(S)    
        train_loader = DataLoader(MyDataset(X, Y), batch_size = batch_size, shuffle = False)

        for _ in range(n_iters):
            loss_vector, Theta_x = compute_model_parameter_fairness(weight, S, ratio_fair, self.model, train_loader, optimizer, num_epochs=num_epochs)
            weight, ratio = compute_weight_with_fairness_no_label(loss_vector, lower, upper, Theta_x, S, cluster)
            weight = torch.from_numpy(weight).float()

        self.weight = weight

    def predict(self, X):
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                out, _ = self.model(X)
            else:
                out, _ = self.model(torch.from_numpy(X).float())
        return torch.max(out.data, dim=1)[1].numpy()

    def eval(self, test_data, test_label, Sen_test):
        test_loader = DataLoader(MyDataset(test_data, test_label), batch_size = len(test_data), shuffle = False)
        Sen_test = torch.from_numpy(Sen_test.reshape(-1, 1)).float()
        self.model.eval()

        running_acc = 0.0
        for data in test_loader:
            img, label = data
            # img = img.view(img.size(0), -1)
            label = label.data.numpy()
            for i in range(len(label)):
                label[i] = 0 if label[i] == 0.0 else 1
            label = torch.LongTensor(label)
            label = label.view(len(label))

            pred = torch.from_numpy(self.predict(img))

            running_acc += torch.sum((pred == label))
            
            pred = torch.where(pred > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            pred = pred.view(len(pred))
            
            count_Y1_S1 = torch.sum(torch.mul(pred, Sen_test))
            count_Y0_S1 = torch.sum(torch.mul(1.0 - pred, Sen_test))
            count_Y1_S0 = torch.sum(torch.mul(pred, 1.0 - Sen_test))
            count_Y0_S0 = torch.sum(torch.mul(1.0 - pred, 1.0 - Sen_test))

            r11 = count_Y1_S1 / len(Sen_test)
            r01 = count_Y0_S1 / len(Sen_test)
            r10 = count_Y1_S0 / len(Sen_test)
            r00 = count_Y0_S0 / len(Sen_test)
            risk_difference = abs(r11 / (r11 + r01) - r10 / (r10 + r00))

        running_acc = running_acc.numpy()
        print('Finish Acc %.6f:' % (running_acc / len(Sen_test)))
        print('Finish risk difference %.6f' % (risk_difference))

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

def compute_weight_with_fairness_no_label(loss_vector, lower_bound, upper_bound, Theta_x, Sen, cluster):
    solvers.options['show_progress'] = False
    Q = 2 * np.zeros((len(cluster), len(cluster)))
    Q = Q.astype('double')

    q_matrix = np.zeros(len(cluster))
    index = 0
    for i in range(len(q_matrix)):
        count = cluster[i]
        for j in range(index, index + count):
            q_matrix[i] += loss_vector[j] / len(loss_vector)
        index = index + count
    q_matrix = q_matrix.reshape(-1, 1)
    q = q_matrix.astype('double')

    ### Compute A, b equality constraint
    A = np.zeros((1, len(cluster)))
    b = [0.0 * len(cluster)]
    A = A.astype('double')

    ### compute fariness constraint
    ### compute first term of fairness constraint
    tau = 0.1
    ### compute fariness constraint
    ### compute first term of fairness constraint

    sen_bar = np.sum(Sen) / len(Sen)

    Theta_x = np.sum(Theta_x, axis=1)
    Theta_x = Theta_x.reshape(len(Theta_x), 1)
    Theta_x = Theta_x / max(Theta_x)

    fair = np.multiply(Sen - sen_bar, Theta_x)
    fair_cons = np.zeros(len(cluster))
    index = 0
    for i in range(len(fair_cons)):
        count = cluster[i]
        for j in range(index, index + count):
            fair_cons[i] += (fair[j] / len(loss_vector))
        index = index + count

    # print("theta")
    # print(Theta_x)

    # print("fair")
    # print(fair_cons)
    fair = fair_cons.reshape(-1, 1)

    # ### G(w, alpha) < tau and G(w, alpha) > -tau
    fair_G = np.concatenate((fair.transpose(), -fair.transpose()))
    fair_h = np.full((2, 1), tau)

    # ### compute G and h inequality constraint and combine with fairness inequality
    lower = -np.identity(len(cluster))
    upper = np.identity(len(cluster))
    G = np.concatenate((lower, upper), axis=0)

    G = np.concatenate((G, fair_G))
    #

    h = np.concatenate((lower_bound, upper_bound), axis=0)
    h = h.reshape(len(h), 1)

    # print("lower bound")
    # print(lower_bound)
    # print(upper_bound)

    h = np.concatenate((h, fair_h), axis=0)
    h = h.astype('double')

    Q, p, G, h, A, b = -matrix(Q), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b)
    #
    sol = solvers.qp(Q, p, G, h)
    sol = np.array(sol['x'])

    # print("solution")
    # print(sol)
    weight = np.zeros(len(Sen))
    index = 0
    max_upper = max(upper_bound)
    min_lower = min(lower_bound)
    for i in range(len(cluster)):
        count = cluster[i]
        for j in range(index, index + count):
            weight[j] = abs(sol[i])
            if weight[j] > max_upper:
                weight[j] = max_upper
            if weight[j] < min_lower:
                weight[j] = min_lower
        index = index + count

    weight = weight.reshape(len(weight), 1)

    ratio = np.sum(weight * Sen) / np.sum(weight)

    return weight, ratio