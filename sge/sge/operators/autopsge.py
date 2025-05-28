import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    torch.set_default_device("mps")
    device = torch.device("mps")
else:
    torch.set_default_device("cpu")
    device = torch.device("cpu")
    
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, non_terminal_slices):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.non_terminal_slices = non_terminal_slices

        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, 128).to(device)
        self.encoder_fc2 = nn.Linear(128, 64).to(device)
        self.fc_mu = nn.Linear(64, latent_dim).to(device)
        self.fc_logvar = nn.Linear(64, latent_dim).to(device)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, 64).to(device)
        self.decoder_fc2 = nn.Linear(64, 128).to(device)
        self.decoder_fc3 = nn.Linear(128, input_dim).to(device)

    def encode(self, x):
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        recon_x = self.decoder_fc3(h)
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


from torch.optim import Adam

def vae_loss(recon_x, x, mu, logvar):
    # TODO: change loss function
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_vae(model, data_loader, epochs=50, lr=1e-3):
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            batch = batch.view(batch.size(0), -1).to(device)  # Flatten input
            optimizer.zero_grad()
            recon_batch, mu, logvar, _ = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader.dataset)}")

    return model

def population_to_vector(population):
    l = []
    for ind in population:
        i = []
        for sublist in ind['genotype']:
            for codon in sublist:
                i.append(codon[1])
        l.append(i)

    vector = torch.tensor(l, dtype=torch.float32).to(device)  # Convert to tensor and move to device
    return vector


def apply_softmax_to_slices(z, non_terminal_slices):
    """
    Apply softmax to each slice of the latent vector.

    Args:
        z (torch.Tensor): Latent vector of shape (batch_size, latent_dim).
        non_terminal_slices (list of int): List where each element is the size of a slice
                                           corresponding to a non-terminal.

    Returns:
        list of torch.Tensor: List of softmax probabilities for each slice.
    """
    start = 0
    softmax_slices = []
    # print(f"Latent vector shape: {z.shape}")
    # input()
    for slice_size in non_terminal_slices:
        end = start + slice_size
        slice_softmax = F.softmax(z[:, start:end], dim=1)  # Apply softmax to the slice
        softmax_slices.append(slice_softmax)
        start = end
    return softmax_slices


def softmax_slices_to_nested_lists(population, softmax_slices):
    """
    Convert softmax slices into a nested list of lists.

    Args:
        softmax_slices (list of torch.Tensor): List of softmax probabilities for each slice.

    Returns:
        list of list of list: Nested list where the outer list corresponds to individuals,
                              the middle list corresponds to non-terminals, and the inner
                              list contains the probability distribution for each non-terminal.
    """
    # Transpose the structure to group by individuals
    num_individuals = softmax_slices[0].shape[0]
    # nested_lists = []

    for i in range(len(population)):
        individual_slices = [slice[i].tolist() for slice in softmax_slices]
        # nested_lists.append(individual_slices)
        population[i]['probabilities'] = individual_slices

    # print(nested_lists)
    # input()
    # return nested_lists


def update_probs(gen, vae, population, non_terminal_slices):
    # print('eval')
    vae.eval()


    if gen % 15 == 0:
        # Convert the population to a vector representation
        population_vector = population_to_vector(population)
        data_loader = torch.utils.data.DataLoader(population_vector, batch_size=32, shuffle=False)
        vae = train_vae(vae, data_loader)  # Train the VAE every 15 iterations
        
        # print('encode')
        # mu, logvar = vae.encode(population_vector)  # Get the mean and log variance from the encoder
        # z = vae.reparameterize(mu, logvar)  # Reparameterize to get the latent vector

    population_vector = population_to_vector([population[0]])
    data_loader = torch.utils.data.DataLoader(population_vector, batch_size=32, shuffle=False)
    mu, logvar = vae.encode(population_vector)  # Get the mean and log variance from the encoder
    z = vae.reparameterize(mu, logvar)


    # recon_x, mu, logvar, z = vae(batch)  # Forward pass to get the latent vector z
        
    # Apply softmax to each slice of the latent vector
    # print('softmax')
    softmax_slices = apply_softmax_to_slices(z, non_terminal_slices)
    # print(softmax_slices)
    # input()
    # Now `softmax_slices` contains the probability distributions for each non-terminal
    # for i, probs in enumerate(softmax_slices):
    #     print(f"Non-terminal {i + 1} probabilities: {probs}")
        # inout()
    # print('vector to pop')

    # FIXME: one prob list per individual
    # softmax_slices_to_nested_lists(population[0], softmax_slices)
    # else: just the first prob array
    probs = [slice[0].tolist() for slice in softmax_slices]

    return vae, probs


def initialize_vae(population, non_terminal_slices):
    # input_dim = flattened genotype
    # latent_dim = size of the latent space, at least the number of rules
    # non_terminal_slices = list of sizes of each non-terminal slice

    ind = population[0] 
    input_dim = sum(len(sublist) for sublist in ind['genotype'])  # Assuming genotype is a list
    # print(f"Input dimension: {input_dim}")
    # print(f"Non-terminal slices: {non_terminal_slices}")
    latent_dim = max(20, sum(non_terminal_slices))  # Ensure latent_dim is at least the size of the largest slice
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, non_terminal_slices=non_terminal_slices) 

    # Convert the population to a vector representation
    population_vector = population_to_vector(population)
    data_loader = torch.utils.data.DataLoader(population_vector, batch_size=32, shuffle=False)

    # Train the VAE
    vae = train_vae(vae, data_loader)

    return vae