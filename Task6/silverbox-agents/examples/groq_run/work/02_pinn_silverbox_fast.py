import os, sys, numpy as np, scipy.io, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# set seeds
torch.manual_seed(0)
np.random.seed(0)

# load data
path = os.getenv('DATA_FILE')
if not path:
    print('DATA_FILE not set')
    sys.exit(1)
mat = scipy.io.loadmat(path)
# find V1, V2 keys
V1 = next(k for k in mat if 'V1' in k)
V2 = next(k for k in mat if 'V2' in k)
u_full = mat[V1].squeeze()
y_full = mat[V2].squeeze()
# slice
start, end = 42650, 46746
u = u_full[start:end]
y = y_full[start:end]
# center
u = u - u.mean()
y = y - y.mean()
# params
dt = 1.0/610.35
N = len(y)
train_len = 3072
test_len = 1024
assert N == 4096
# velocity via central diff
v = np.gradient(y, dt)
# scaling std on training part
y_std = y[:train_len].std()
v_std = v[:train_len].std()
u_std = u[:train_len].std()

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class AccelNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, 128))
        layers.append(Swish())
        for _ in range(3):
            layers.append(nn.Linear(128, 128))
            layers.append(Swish())
        layers.append(nn.Linear(128, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer('scale', torch.tensor([y_std, v_std, u_std], dtype=torch.float32))
    def forward(self, y, v, u):
        inp = torch.stack([y, v, u], dim=-1) / self.scale
        return self.net(inp).squeeze(-1)

net = AccelNet().float()

# RK4 step
def rk4_step(y, v, u, net):
    dt_t = torch.tensor(dt, dtype=torch.float32)
    def f(y_, v_, u_):
        return net(y_, v_, u_)
    k1y = v
    k1v = f(y, v, u)
    k2y = v + 0.5*dt_t*k1v
    k2v = f(y + 0.5*dt_t*k1y, v + 0.5*dt_t*k1v, u)
    k3y = v + 0.5*dt_t*k2v
    k3v = f(y + 0.5*dt_t*k2y, v + 0.5*dt_t*k2v, u)
    k4y = v + dt_t*k3v
    k4v = f(y + dt_t*k3y, v + dt_t*k3v, u)
    y_next = y + (dt_t/6)*(k1y + 2*k2y + 2*k3y + k4y)
    v_next = v + (dt_t/6)*(k1v + 2*k2v + 2*k3v + k4v)
    return y_next, v_next

# Dataset for rollouts
class RolloutDataset(Dataset):
    def __init__(self, y, v, u, L, max_start, size):
        self.y = y
        self.v = v
        self.u = u
        self.L = L
        self.max_start = max_start
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        start = np.random.randint(0, self.max_start+1)
        return (torch.from_numpy(self.y[start:start+self.L].astype(np.float32)),
                torch.from_numpy(self.v[start:start+self.L].astype(np.float32)),
                torch.from_numpy(self.u[start:start+self.L].astype(np.float32)))

L = 128
max_start = train_len - L
train_dataset = RolloutDataset(y, v, u, L, max_start, size=200)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

for epoch in range(30):
    for y_batch, v_batch, u_batch in train_loader:
        B, Lb = y_batch.shape
        y0 = y_batch[:,0]
        v0 = v_batch[:,0]
        y_pred = y0
        v_pred = v0
        for k in range(Lb):
            u_k = u_batch[:,k]
            y_pred, v_pred = rk4_step(y_pred, v_pred, u_k, net)
        # loss on final point vs true
        loss = ((y_pred - y_batch[:,-1])**2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    if (epoch+1)%5==0:
        print(f'Epoch {epoch+1} loss {loss.item():.6f}')

# free run
net.eval()
with torch.no_grad():
    y_cur = torch.tensor(y[0], dtype=torch.float32)
    v_cur = torch.tensor((y[1]-y[0])/dt, dtype=torch.float32)
    preds = []
    for k in range(N):
        u_k = torch.tensor(u[k], dtype=torch.float32)
        y_cur, v_cur = rk4_step(y_cur, v_cur, u_k, net)
        preds.append(y_cur.item())
    preds = np.array(preds)

train_rmse = np.sqrt(((preds[:train_len] - y[:train_len])**2).mean())
test_rmse = np.sqrt(((preds[train_len:] - y[train_len:])**2).mean())
print(f'TRAIN_RMSE={train_rmse}')
print(f'TEST_RMSE={test_rmse}')
