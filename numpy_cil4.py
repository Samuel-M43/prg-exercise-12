import numpy as np

# --- PŘÍPRAVA DAT (Spusť pouze pokud soubor ještě nemáš na disku) ---
# Vytvoříme testovací data: 20 pacientů, 3 sloupce (systol, diastol, tep)
# np.random.seed(42)
# test_data = np.random.randint([110, 70, 60], [160, 100, 90], size=(20, 3))
# np.save("pacienti.npy", test_data)

# --- Načtení souboru a výpis tvaru ---
data = np.load("pacienti.npy")
print(f"Tvar pole: {data.shape}")
print(f"Prvych 5 riadkov: \n {data[:5]}")


# --- Průměr a směrodatná odchylka ---
mean_vals = np.mean(data, axis=0)
std_vals = np.std(data, axis=0)

print(f"Priemery: {mean_vals}")
print(f"Odchylky: {std_vals}")

# --- Hypertenzia ---
hyper = data[data[:,0] >= 140]
print(f"Pocet pacientov s hypertenziou: {hyper.shape[0]}")
print(f"Priemerny systolicky tlak: {hyper[:,0].mean()}")


pulse_pressure = data[:, 0] - data[:, 1]
data_extended = np.column_stack([data, pulse_pressure])

print("Nový tvar dat:", data_extended.shape)
print("Prvych 5 riadkov s pulznym tlakom:")
print(data_extended[:5])

idx = np.argmax(data[:, 2])

print("Index pacienta s najvyssim TF:", idx)
print("Jeho záznam:", data[idx])

