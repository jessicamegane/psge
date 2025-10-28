import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sge.parameters import params


if torch.cuda.is_available():
    torch.set_default_device("cuda")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    torch.set_default_device("mps")
    device = torch.device("mps")
else:
    torch.set_default_device("cpu")
    device = torch.device("cpu")
    
# BATCH_SIZE = 64
# TRAIN_INTERVAL = 25
# EPOCH = 100
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim = 512, second_hidden_dim = 128):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.LeakyReLU(0.1),
        )

        self.mean_layer = nn.Linear(second_hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(second_hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, second_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(second_hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),   # OK because my data is normalized to [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


from torch.optim import Adam, SGD, NAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
#, CosineAnnealingWarmRestarts

def vae_loss(recon_x, x, mu, logvar):
    # TODO: change loss function
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def vae_loss_fitness(recon_x, x, mu, logvar, fitness):
    recon_loss = F.mse_loss(recon_x, x, reduction='none').sum(dim=-1)  # Shape: [batch_size]
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # Shape: [batch_size]
    
    # print(f"recon_loss shape: {recon_loss.shape}, kl_loss shape: {kl_loss.shape}, fitness shape: {fitness.shape}")
    # print(f"recon_loss mean: {recon_loss.mean()}, kl_loss mean: {kl_loss.mean()}")
    
    total_loss = (recon_loss + kl_loss) * fitness
    # print(f"Total loss shape: {total_loss.shape}, Total loss mean: {total_loss.mean()}")
    # print(f"Total loss sum: {total_loss.sum()}")
    # input()
    
    return total_loss.sum()  # or total_loss.mean() depending on your preference


def train_vae(model, data_loader, epochs=params['EPOCHS'], lr=1e-3, fitness=False):
    # optimizer = NAdam(model.parameters(), lr=lr)
    optimizer = model.optimizer
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_num = 0
        for batch in data_loader:
            if fitness:
                batch_inputs, batch_targets = batch
            else:
                batch_inputs = batch
            batch_inputs = batch_inputs.view(batch_inputs.size(0), -1).to(device)  # Flatten input
            optimizer.zero_grad()
            recon_batch, mu, logvar, _ = model(batch_inputs)
            if fitness:
                loss = vae_loss_fitness(recon_batch, batch_inputs, mu, logvar, batch_targets)
            else:
                loss = vae_loss(recon_batch, batch_inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader.dataset)}")
        model.scheduler.step() 
    return model

def decode_pop(vae, population):

    population_vector = population_to_vector(population)
    # data_loader = torch.utils.data.DataLoader(population_vector, batch_size=params['BATCH_SIZE'], shuffle=False)

    decoded_population = []
    mu, std = vae.encode(population_vector)  # Get the mean and log variance from the encoder
    z = vae.reparameterize(mu, std)  # Reparameterize to get the latent vector

    decoded_population = vae.decode(z)  # Decode the latent vector to get the reconstructed population
    # print(decoded_population)

    # with torch.no_grad():
    #     for batch in data_loader:
    #         recon_batch, _, _, _ = vae(batch.to(device))
    #         decoded_population.extend(recon_batch.cpu().numpy())

    new_population = vector_to_population(decoded_population, population)

    return new_population

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

def population_to_vector_with_fitness(population):
    inputs = []
    targets = []
    for ind in population:
        i = []
        for sublist in ind['genotype']:
            for codon in sublist:
                i.append(codon[1])
        inputs.append(i)
        targets.append(ind['fitness'])

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    return inputs_tensor, targets_tensor


def train_population_to_vector(population):
    l = []
    print(len(population))
    for ind in population:
        i = []
        for sublist in ind:
            for codon in sublist:
                i.append(codon[1])
        l.append(i)
    vector = torch.tensor(l, dtype=torch.float32).to(device)  # Convert to tensor and move to device
    return vector

def vector_to_population_batch(vectors, original_population_structure):
    # print(len(vectors))
    # print(vectors[0].shape)
    population = []
    i = 0
    for batch in vectors:
        for recon_individual in batch:
            new_individual = original_population_structure[i].copy()
            new_genotype = []
            start = 0
            for j, nt in enumerate(new_individual['genotype']):
                end = start + len(nt)
                nt_list = []
                for k in range(start, end):
                    nt_list.append([-1, recon_individual[k].item(), -1])
                new_genotype.append(nt_list)
                start = end

            new_individual['genotype'] = new_genotype
            new_individual['fitness'] = None  # Reset fitness
            new_individual['tree_depth'] = None  # Reset tree depth
            i += 1
            population.append(new_individual)
    return population


def vector_to_population(vectors, original_population_structure):
    # print(len(vectors))
    # print(vectors[0].shape)
    population = []
    i = 0
    for recon_individual in vectors:
        new_individual = original_population_structure[i].copy()
        new_genotype = []
        start = 0
        for j, nt in enumerate(new_individual['genotype']):
            end = start + len(nt)
            nt_list = []
            for k in range(start, end):
                nt_list.append([-1, recon_individual[k].item(), -1])
            new_genotype.append(nt_list)
            start = end

        new_individual['genotype'] = new_genotype
        new_individual['fitness'] = None  # Reset fitness
        new_individual['tree_depth'] = None  # Reset tree depth
        i += 1
        population.append(new_individual)
    return population
   

def apply_softmax_to_slices(z, non_terminal_slices):
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

def calculate_pooled_mean_std(mean_array, std_array):

    # Calculate pooled mean
    pooled_mean = torch.mean(mean_array, dim=0)
    # Calculate pooled standard deviation
    pooled_std = torch.sqrt(torch.mean(std_array**2, dim=0) + torch.mean(mean_array**2, dim=0) - pooled_mean**2)

    return pooled_mean, pooled_std

def change_probs_mean_std(probs, mean, std, non_terminal_slices):
    # print("update with mean and std attempt")
    # print(f"mean: {mean}")
    # print(f"std: {std}")
    # mean, std = calculate_pooled_mean_std(mean, std)

    # apply softmax to each slice of the latent vector
    mean_softmax = apply_softmax_to_slices(mean, non_terminal_slices)
    std_softmax = apply_softmax_to_slices(std, non_terminal_slices)
    # print(f"mean_softmax: {mean_softmax}")


    # print(mean_softmax)
    # input()


    # print(f"mean: {mean}")
    # print(f"std: {std}")
    # print(probs)
    # print(f"non_terminal_slices: {non_terminal_slices}")
    i = 0
    ii = 0
    # non-softmax
    # sample = np.random.normal(mean.tolist(), np.abs(std.tolist())).tolist()[0]
    # print(sample)
    for nt in probs:
        j = non_terminal_slices[i]
        mean = mean_softmax[i].tolist()[0]
        std = std_softmax[i].tolist()[0]
        sample = np.random.normal(mean[:j], np.abs(std[:j]))


        # PREVIOUS
        nt[:j] = nt[:j] + sample
        # NON_SOFTMAX
        # nt[:j] = nt[:j] + sample[ii:ii+j]

        nt[:j] = F.softmax(torch.tensor(nt[:j]), dim=0).tolist()
        i += 1
        ii = j
    # print(probs)
    # input()
    return probs

def separate_fitness_by_batch(population, batch_size):
    batches = []
    for i in range(0, len(population), batch_size):
        batch = [indiv['fitness'] for indiv in population[i:i + batch_size]]
        batches.append(batch)
    return batches

def update_probs(gen, probs, vae, population, non_terminal_slices):
    # print('eval')
    vae.eval()


    if gen % params['TRAIN_INTERVAL'] == 0 and gen != 0:
        # Convert the population to a vector representation
        population_vector = population_to_vector(population)

        # fitness_vector = separate_fitness_by_batch(population, params['BATCH_SIZE'])
        # fitness_vector = torch.tensor(fitness_vector, dtype=torch.float32).to(device) 
        
        # data_loader = torch.utils.data.DataLoader(population_vector, batch_size=params['BATCH_SIZE'], shuffle=False)
        inputs_vector, targets_vector = population_to_vector_with_fitness(population)

        dataset = torch.utils.data.TensorDataset(inputs_vector, targets_vector)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        
        vae = train_vae(vae, data_loader, fitness = True)  # Train the VAE every 15 iterations


    population_vector = population_to_vector([population[0]])  # Convert only the first individual to a vector representation
    data_loader = torch.utils.data.DataLoader(population_vector, batch_size=params['BATCH_SIZE'], shuffle=False)
    mu, logvar = vae.encode(population_vector)  # Get the mean and log variance from the encoder
    # z = vae.reparameterize(mu, logvar)
    

    # recon_x, mu, logvar, z = vae(batch)  # Forward pass to get the latent vector z
        
    # Apply softmax to each slice of the latent vector
    # print('softmax')
    # softmax_slices = apply_softmax_to_slices(z, non_terminal_slices)
    # print(softmax_slices)
    # input()
    # Now `softmax_slices` contains the probability distributions for each non-terminal
    # for i, probs in enumerate(softmax_slices):
    #     print(f"Non-terminal {i + 1} probabilities: {probs}")
        # inout()
    # print('vector to pop')

    # NOTE: one prob list per individual
    # softmax_slices_to_nested_lists(population[0], softmax_slices)
    # else: just the first prob array
    # probs = [slice[0].tolist() for slice in softmax_slices]
    # print(probs)
    probs = change_probs_mean_std(probs, mu, logvar, non_terminal_slices)
    # print(probs)
    # input()
    return vae, probs


def initialize_vae(population, non_terminal_slices, pop):
    # input_dim = flattened genotype
    # latent_dim = size of the latent space, at least the number of rules
    # non_terminal_slices = list of sizes of each non-terminal slice

    ind = pop[0] 
    input_dim = sum(len(sublist) for sublist in ind['genotype'])  # Assuming genotype is a list
    # print(f"Input dimension: {input_dim}")
    # print(f"Non-terminal slices: {non_terminal_slices}")
    # latent_dim = max(20, sum(non_terminal_slices))  # Ensure latent_dim is at least the size of the largest slice
    latent_dim = sum(non_terminal_slices)
    # print("Latent dimension:", latent_dim)
    # print("Input dimension:", input_dim)
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim) 
    vae.to(device)  # Move the VAE to the appropriate device

    # Convert the population to a vector representation
    population_vector = train_population_to_vector(population)
    data_loader = torch.utils.data.DataLoader(population_vector, batch_size=params['BATCH_SIZE'], shuffle=False)

    # Train the VAE
    optimizer = NAdam(vae.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=params['EPOCHS'], eta_min=1e-6)
    vae.optimizer = optimizer  
    vae.scheduler = scheduler
    vae = train_vae(vae, data_loader)

    return vae