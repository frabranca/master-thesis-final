import yaml

# Load the YAML file
lif = 'results_inference/31f8f8028de64207aad08ecf3d596897/metrics_68.yml'
rnn_2_B = 'results_inference/128d8bb90e4e4b8fbc3123471ba0b6ca/metrics_1.yml'
if_2_B = 'results_inference/128d8bb90e4e4b8fbc3123471ba0b6ca/metrics_4.yml'
rnn_2_A = 'results_inference/1f5735184f0a47abb20b0409c515b4c1/metrics_3.yml'
if_2_A = 'results_inference/1f5735184f0a47abb20b0409c515b4c1/metrics_6.yml'
rnn_3_A = 'results_inference/0e045bd7fce64ac488134591252a696a/metrics_3.yml'

with open(rnn_3_A, 'r') as file:
    data = yaml.safe_load(file)

# Get the AEE and RSAT values
aee_values = data['AEE']
rsat_values = data['RSAT']

aee_list = [float(value) for value in list(aee_values.values())]
rsat_list = [float(value) for value in list(rsat_values.values())]

# Calculate the average
aee_average = sum(aee_list) / len(aee_list)
rsat_average = sum(rsat_list) / len(rsat_list)

# Print the averages
print(f"AEE Average: {aee_average}")
print(f"RSAT Average: {rsat_average}")