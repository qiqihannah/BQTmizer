import pandas as pd
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import random
import time
import numpy as np
import dwave.inspector
from dwave.cloud.exceptions import SolverError

class Sampler:
    def __init__(self, data_len, beta, subproblem_size):
        self.data_len = data_len
        self.beta = beta
        self.subproblem_size = subproblem_size

    def random_sample_unique(self):
        count_list = []
        for i in range(10):
            count = 0
            selected_samples = random.sample(range(self.data_len), self.subproblem_size)
            while len(selected_samples) <= self.data_len * self.beta:
                sample = random.sample(range(self.data_len), self.subproblem_size)
                selected_samples += sample
                selected_samples = list(set(selected_samples))
                count += 1
            count_list.append(count)

        return int(np.mean(count_list))

    def bootstrap_sampling(self, sample_time):
        '''
        param data: data frame
        :param sample_time: number of sampling
        :param sample_size: number of test cases in a sample
        :return: a two-dimension list of all sampled test cases
        '''
        sample_list_total = []
        for i in range(sample_time):
            sample_list = random.sample(range(self.data_len), self.subproblem_size)
            sample_list_total.append(sample_list)
        return sample_list_total


class QA_solver:
    def __init__(self, subproblem_size, data, eff_features, cost_features, weights):
        self.subproblem_size = subproblem_size
        self.data = data
        self.eff_features = eff_features
        self.cost_features = cost_features
        self.weights = weights

    def create_bqm(self, sample): # feature list
        '''
        :param sample: a list of sampled test cases
        :param sample_time:
        :param data: dataframe
        :return: a bqm of the objective function
        '''
        dic_num = {}
        bqm_total = 0

        eff_total = 0
        for eff in self.eff_features:
            dic_eff = {}
            for id in sample:
                dic_eff["T"+str(id)] = self.data[eff].iloc[id]
                eff_total += self.data[eff].iloc[id]
                bqm_eff = dimod.BinaryQuadraticModel(dic_eff,{}, 0,dimod.Vartype.BINARY)
            if any(x > 1.0 or x < 0.0 for x in self.data[eff]):
                bqm_eff.normalize()
            bqm_total += self.weights[eff]*pow((bqm_eff-eff_total) / self.subproblem_size, 2)
        cost_total = 0
        for cost in self.cost_features:
            dic_cost = {}
            for id in sample:
                dic_cost["T" + str(id)] = self.data[cost].iloc[id]
                cost_total += self.data[cost].iloc[id]
                bqm_cost = dimod.BinaryQuadraticModel(dic_cost, {}, 0, dimod.Vartype.BINARY)
            if any(x > 1.0 or x < 0.0 for x in self.data[cost]):
                bqm_cost.normalize()
            bqm_total += self.weights[cost] * pow((bqm_cost) / self.subproblem_size, 2)
        for id in sample:
            dic_num["T"+str(id)] = 1
        bqm_num = dimod.BinaryQuadraticModel(dic_num, {}, 0, dimod.Vartype.BINARY)
        bqm_total += self.weights["num"] * pow((bqm_num - 0) / self.subproblem_size, 2)
        return bqm_total

    def run_qpu(self, sample_list_total, sample_size, token):
        '''
        :param sample_list_total: all sampled test cases
        :param data: dataframe
        :return: energy and sample of the best solution
        '''
        sample_first_list = []
        energy_first_list = []
        execution_time = 0
        execution_time_list = []
        sampleset_list = []
        max_qubit = 0
        qpu_access_list = []
        qubit_num_list = []

        for i in range(len(sample_list_total)):

            # Update the status message for the current sub-problem
            print(f"Processing sub-problem {i + 1}/{len(sample_list_total)}...")
            obj = self.create_bqm(sample_list_total[i])

            attempt = 0
            retries = 5
            while attempt < retries:
                try:
                    start = time.time()
                    sampler = EmbeddingComposite(DWaveSampler(token=token))
                    embedding = time.time()
                    sampleset = sampler.sample(obj, num_reads=100)
                    end = time.time()
                    sampling_time = end - embedding
                    spent_time = end - start
                    qpu_access = sampleset.info['timing']['qpu_access_time']
                    embedding_time = spent_time - qpu_access
                    execution_time += spent_time
                    execution_time_list.append(spent_time)
                    print("success")
                    break
                except (SolverError, ConnectionError, KeyError) as e:
                    # Handle the error and retry after waiting
                    attempt += 1
                    print(f"Attempt {attempt} failed with error: {e}")

                    if attempt < retries:
                        print(f"Pausing for {20} seconds before retrying...")
                        time.sleep(20)  # Pause before retrying
                    else:
                        print("Max retries reached. Could not complete the job.")
                        raise e  # Raise the exception if max retries are exceeded

            first_sample = sampleset.first.sample
            first_energy = sampleset.first.energy
            sample_first_list.append(first_sample)
            energy_first_list.append(first_energy)
            qpu_access_list.append(qpu_access)

            sample_list = sample_list_total[i]
            selected_list = [int(x) for x in [list(first_sample.keys())[id][1:] for id in range(sample_size) if
                                              list(first_sample.values())[id] == 1]]
            selected_num = len(selected_list)
            fval = first_energy
            embedding = sampleset.info['embedding_context']['embedding']

            qubit_num = sum(len(chain) for chain in embedding.values())
            qubit_num_list.append(qubit_num)

            if i == 0:
                head_df = ["sample_list", "selected_list", "selected_num", "fval", "qubit_num", "spent_time(s)",
                           "embedding_time(s)", "sampling_time(s)"]
                head_df += list(sampleset.info.keys())
                df_log = pd.DataFrame(columns=head_df)
            values_df = [sample_list, selected_list, selected_num, fval, qubit_num, spent_time, embedding_time,
                         sampling_time]
            values_df += list(sampleset.info.values())
            df_log.loc[len(df_log)] = values_df
            sampleset_list.append(sampleset.to_pandas_dataframe())

            max_qubit += sum(len(chain) for chain in embedding.values())
            print(f"{max_qubit} physical qubits have been used in total.")

        return sample_first_list, qpu_access_list, sum(qubit_num_list) / len(qubit_num_list), df_log, sampleset_list

class ResultBuilder:
    def __init__(self, sample_list, data, cost_feature, eff_feature, weights):
        self.data = data
        self.cost_feature = cost_feature
        self.eff_feature = eff_feature
        self.weights = weights
        self.sample_list = sample_list

    def merge(self):
        case_list = {}
        for i in range(len(self.sample_list)):
            for t in self.sample_list[i].keys():
                if t[0] == 'T' and self.sample_list[i][t] == 1:
                    case_list[t] = 1
        return case_list

    def print_results(self, sample, qpu_access_list, qubit_avg):
        features = self.eff_features + self.cost_features
        result_columns = ["index"]
        result_columns.extend(features)
        result_df = pd.DataFrame(columns=result_columns)
        count = 0
        for t in sample.keys():
            if t[0] == 'T' and sample[t] == 1:
                result_list = [t[1:]]
                for feature in features:
                    result_list.append(self.data[feature][int(t[1:])])
                result_df.loc[len(result_df)] = result_list
                count += 1
        fval_total = 0
        for feature in self.eff_features:
            if any(x > 1.0 or x < 0.0 for x in self.data[feature]):
                feature_values_n = [self.data[feature][index] / max(self.data[feature]) for index in
                                       range(len(self.data))]
                feature_sel_n = [result_df[feature][index] / max(self.data[feature]) for index in range(len(result_df))]
                fval_total += self.weights[feature]*pow((sum(feature_sel_n)-sum(feature_values_n))/len(self.data), 2)
            else:
                fval_total += self.weights[feature]*pow((sum(result_df[feature])-sum(self.data[feature]))/len(self.data), 2)
        for feature in self.cost_features:
            if any(x > 1.0 or x < 0.0 for x in self.data[feature]):
                feature_sel_n = [result_df[feature][index] / max(self.data[feature]) for index in range(len(result_df))]
                fval_total += self.weights[feature] * pow(sum(feature_sel_n) / len(self.data), 2)
            else:
                fval_total += self.weights[feature] * pow(sum(result_df[feature]) / len(self.data), 2)
        fval_total += self.weights["num"]*pow(count/len(self.data),2)

        sum_list = [count]
        sum_headers = ["selected_case_num"]
        for feature in features:
            sum_list.append(sum(result_df[feature]))
            sum_headers.append("total_"+feature)
        sum_list+=[fval_total, round(sum(qpu_access_list)*pow(10, -6),3), round(qubit_avg,2)]
        sum_headers+=["fval", "qpu_access_time", "avg_qubit_num"]
        sum_df = pd.DataFrame(columns=sum_headers)
        sum_df.loc[len(sum_df)] = sum_list
        return result_df, sum_df