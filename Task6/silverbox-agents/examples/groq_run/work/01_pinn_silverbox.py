import os, sys, math, numpy as np, scipy.io
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load data
data_path = os.getenv('DATA_FILE')
if not data_path:
    print('DATA_FILE env not set')
    sys.exit(1)
mat = scipy.io.loadmat(data_path)
# Assume channels named 'V1' and 'V2'
# Find keys containing V1 and V2
keys = list(mat.keys())
V1_key = next((k for k in keys if 'V1' in k), None)
V2_key = next((k for k in keys if 'V2' in k), None)
if V1_key is None or V2_key is None:
    print('Channel keys not found')
    sys.exit(1)
u_full = mat[V1_key].squeeze()
y_full = mat[V2_key].squeeze()
# slice
start_idx = 42650
end_idx = 46746  # exclusive
u = u_full[start_idx:end_idx]
y = y_full[start_idx:end_idx]
# center
u = u - u.mean()
y = y - y.mean()
# parameters
dt = 1.0/610.35
N = len(y)
train_len = 3072
test_len = 1024
assert N == 4096
# compute velocity via central diff on whole slice
v = np.gradient(y, dt)
# scaling stds on training part
y_std = y[:train_len].std()
v_std = v[:train_len].std()
u_std = u[:train_len].std()
# MLP architecture B: 4 hidden layers of 128, swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class AccelNet(nn.Module):
    def __init__(self, in_dim=3, hidden=128, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(Swish())
        for _ in range(depth-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(Swish())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        # scaling factors
        self.register_buffer('scale', torch.tensor([y_std, v_std, u_std], dtype=torch.float32))
    def forward(self, y, v, u):
        inp = torch.stack([y, v, u], dim=-1) / self.scale
        return self.net(inp).squeeze(-1)

net = AccelNet()
net = net.float()

# RK4 step
def rk4_step(y, v, u, net):
    # y, v, u are tensors scalar
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
    y_next = y + (dt_t/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
    v_next = v + (dt_t/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    return y_next, v_next

# Dataset for rollouts
class RolloutDataset(Dataset):
    def __init__(self, y, v, u, L, max_start):
        self.y = y
        self.v = v
        self.u = u
        self.L = L
        self.max_start = max_start
    def __len__(self):
        return 100000  # large virtual size
    def __getitem__(self, idx):
        # random start
        start = np.random.randint(0, self.max_start+1)
        return (torch.from_numpy(self.y[start:start+self.L].astype(np.float32)),
                torch.from_numpy(self.v[start:start+self.L].astype(np.float32)),
                torch.from_numpy(self.u[start:start+self.L].astype(np.float32)))

L = 256
max_start = train_len - L
train_dataset = RolloutDataset(y, v, u, L, max_start)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
# cosine scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

lambda_data = 1.0
lambda_phys = 0.0  # ignore physics term for simplicity

for epoch in range(100):
    for batch in train_loader:
        y_batch, v_batch, u_batch = batch  # each shape [B, L]
        B, Lb = y_batch.shape
        # initial state from data
        y0 = y_batch[:,0]
        v0 = v_batch[:,0]
        y_pred = y0
        v_pred = v0
        # integrate over rollout
        for k in range(Lb):
            u_k = u_batch[:,k]
            y_pred, v_pred = rk4_step(y_pred, v_pred, u_k, net)
        # loss at start and end
        loss_start = ((y_pred - y_batch[:,-1])**2).mean()
        loss = lambda_data * loss_start
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
    if (epoch+1)%20==0:
        print(f'Epoch {epoch+1} loss {loss.item():.6f}')

# Free run over whole record
net.eval()
with torch.no_grad():
    y0 = torch.tensor(y[0], dtype=torch.float32)
    # finite diff for v0 using first two points
    v0 = torch.tensor((y[1]-y[0])/dt, dtype=torch.float32)
    y_pred_all = []
    y_cur = y0
    v_cur = v0
    for k in range(N):
        u_k = torch.tensor(u[k], dtype=torch.float32)
        y_cur, v_cur = rk4_step(y_cur, v_cur, u_k, net)
        y_pred_all.append(y_cur.item())
    y_pred_all = np.array(y_pred_all)

# compute RMSE
train_rmse = np.sqrt(((y_pred_all[:train_len] - y[:train_len])**2).mean())
test_rmse = np.sqrt(((y_pred_all[train_len:] - y[train_len:])**2).mean())
print(f'TRAIN_RMSE={train_rmse}')
print(f'TEST_RMSE={test_rmse}')
