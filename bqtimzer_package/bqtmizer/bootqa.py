import pandas as pd
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import random
import time
import numpy as np
import dwave.inspector
from dwave.cloud.exceptions import SolverError

def get_data(data_name):
    data = pd.read_csv("code/dataset/"+data_name+".csv", dtype={"time": float, "rate": float})
    data = data.drop(data[data['rate'] == 0].index)
    return data

def bootstrap_sampling(data_len, sample_time, sample_size):
    '''
    param data: data frame
    :param sample_time: number of sampling
    :param sample_size: number of test cases in a sample
    :return: a two-dimension list of all sampled test cases
    '''
    sample_list_total = []
    for i in range(sample_time):
        sample_list = random.sample(range(data_len), sample_size)
        sample_list_total.append(sample_list)
    return sample_list_total


def random_sample_unique(data_len, cover, subproblem_size):
    count_list = []
    for i in range(10):
        count = 0
        # Sample m times, each sample of size n
        selected_samples = random.sample(range(data_len), subproblem_size)
        # print(selected_samples)
        while len(selected_samples) <= data_len * cover:
            sample = random.sample(range(data_len), subproblem_size)
            selected_samples += sample
            # print(selected_samples)
            selected_samples = list(set(selected_samples))
            # Get unique selected elements
            count += 1
        count_list.append(count)

    return int(np.mean(count_list))


def create_bqm(sample, sample_size, data, eff_features, cost_features, weights): # feature list
    '''
    :param sample: a list of sampled test cases
    :param sample_time:
    :param data: dataframe
    :return: a bqm of the objective function
    '''
    dic_num = {}
    bqm_total = 0

    eff_total = 0
    for eff in eff_features:
        dic_eff = {}
        for id in sample:
            dic_eff["T"+str(id)] = data[eff].iloc[id]
            eff_total += data[eff].iloc[id]
            bqm_eff = dimod.BinaryQuadraticModel(dic_eff,{}, 0,dimod.Vartype.BINARY)
        if any(x > 1.0 or x < 0.0 for x in data[eff]):
            bqm_eff.normalize()
        bqm_total += weights[eff]*pow((bqm_eff-eff_total)/sample_size, 2)
    cost_total = 0
    for cost in cost_features:
        dic_cost = {}
        for id in sample:
            dic_cost["T" + str(id)] = data[cost].iloc[id]
            cost_total += data[cost].iloc[id]
            bqm_cost = dimod.BinaryQuadraticModel(dic_cost, {}, 0, dimod.Vartype.BINARY)
        if any(x > 1.0 or x < 0.0 for x in data[cost]):
            bqm_cost.normalize()
        # bqm_cost.normalize()
        bqm_total += weights[cost] * pow((bqm_cost) / sample_size, 2)
    for id in sample:
        dic_num["T"+str(id)] = 1
    bqm_num = dimod.BinaryQuadraticModel(dic_num, {}, 0, dimod.Vartype.BINARY)
    bqm_total += weights["num"] * pow((bqm_num - 0) / sample_size, 2)
    return bqm_total


def run_qpu(sample_list_total, data, sample_size, data_name, token, eff_features, cost_features, weights):
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

    # progress_bar = st.progress(0)
    # status_text = st.empty()
    # qubit_text = st.empty()
    for i in range(len(sample_list_total)):

        # Update the status message for the current sub-problem
        print(f"Processing sub-problem {i + 1}/{len(sample_list_total)}...")
        obj = create_bqm(sample_list_total[i], sample_size, data, eff_features, cost_features, weights)

        attempt = 0
        retries = 5
        while attempt < retries:
            try:
                start = time.time()
                sampler = EmbeddingComposite(DWaveSampler(token=token))
                embedding = time.time()
                sampleset = sampler.sample(obj, num_reads=100)
                # Formatter(sorted_by=None).fprint(sampleset)
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
        selected_list = [int(x) for x in [list(first_sample.keys())[id][1:] for id in range(sample_size) if list(first_sample.values())[id] == 1]]
        selected_num = len(selected_list)
        fval = first_energy
        embedding = sampleset.info['embedding_context']['embedding']

        qubit_num = sum(len(chain) for chain in embedding.values())
        qubit_num_list.append(qubit_num)

        if i == 0:
            head_df = ["sample_list", "selected_list", "selected_num", "fval", "qubit_num", "spent_time(s)", "embedding_time(s)", "sampling_time(s)"]
            head_df += list(sampleset.info.keys())
            df_log = pd.DataFrame(columns=head_df)
        values_df = [sample_list, selected_list, selected_num, fval, qubit_num, spent_time, embedding_time, sampling_time]
        values_df += list(sampleset.info.values())
        df_log.loc[len(df_log)] = values_df
        sampleset_list.append(sampleset.to_pandas_dataframe())

        # print(f"Number of logical variables: {len(embedding.keys())}")
        # print(f"Number of physical qubits used in embedding: {sum(len(chain) for chain in embedding.values())}")
        #
        # # st.write(f"Sub-problem {i + 1}/{len(sample_list_total)} complete.")
        # # st.write(f"{sum(len(chain) for chain in embedding.values())} physical qubits used in embedding.")
        max_qubit += sum(len(chain) for chain in embedding.values())
        print(f"{max_qubit} physical qubits have been used in total.")
        # progress_bar.progress((i + 1) / len(sample_list_total))

    return sample_first_list, qpu_access_list, sum(qubit_num_list)/len(qubit_num_list), df_log, sampleset_list
    # return sample_first_list, energy_first_list, execution_time, max(execution_time_list), df_log, sampleset_list

def gen_dic(data):
    foods = {}
    for i,x in enumerate(data[["time","rate"]].to_dict(orient="records")):
        foods["T{}".format(i)] = x
    return foods

# sample_first_list, qpu_access_list, sum(qubit_num_list)/len(qubit_num_list)
def print_results(sample, data, qpu_access_list, qubit_avg, cost_features, eff_features,weights):
    features = eff_features + cost_features
    result_columns = ["index"]
    result_columns.extend(features)
    result_df = pd.DataFrame(columns=result_columns)
    count = 0
    for t in sample.keys():
        if t[0] == 'T' and sample[t] == 1:
            result_list = [t[1:]]
            for feature in features:
                result_list.append(data[feature][int(t[1:])])
            result_df.loc[len(result_df)] = result_list
            count += 1
    fval_total = 0
    for feature in eff_features:
        if any(x > 1.0 or x < 0.0 for x in data[feature]):
            feature_values_n = [data[feature][index] / max(data[feature]) for index in
                                   range(len(data))]
            feature_sel_n = [result_df[feature][index] / max(data[feature]) for index in range(len(result_df))]
            fval_total += weights[feature]*pow((sum(feature_sel_n)-sum(feature_values_n))/len(data), 2)
        else:
            fval_total += weights[feature]*pow((sum(result_df[feature])-sum(data[feature]))/len(data), 2)
    for feature in cost_features:
        if any(x > 1.0 or x < 0.0 for x in data[feature]):
            feature_sel_n = [result_df[feature][index] / max(data[feature]) for index in range(len(result_df))]
            fval_total += weights[feature] * pow(sum(feature_sel_n) / len(data), 2)
        else:
            fval_total += weights[feature] * pow(sum(result_df[feature]) / len(data), 2)
    fval_total += weights["num"]*pow(count/len(data),2)

    sum_list = [count]
    sum_headers = ["selected_case_num"]
    for feature in features:
        sum_list.append(sum(result_df[feature]))
        sum_headers.append("total_"+feature)
    sum_list+=[fval_total, round(sum(qpu_access_list)*pow(10, -6),3), round(qubit_avg,2)]
    sum_headers+=["fval", "qpu_access_time", "avg_qubit_num"]
    sum_df = pd.DataFrame(columns=sum_headers)
    sum_df.loc[len(sum_df)] = sum_list

    # result_df.to_csv("BootQA1/"+data_name+"/size_"+str(sample_size)+"/result.csv")
    # sum_df.to_csv("BootQA1/" + data_name + "/size_" + str(sample_size) + "/sum.csv")
    return result_df, sum_df



def print_diet(sample,data, data_name, sample_size, execution_time, max_time, cost_features, eff_features,weights):
    result_columns = ["index"]
    result_columns.extend(data.columns[1:])
    print(result_columns)
    result_df = pd.DataFrame(columns=result_columns)
    feature_values = gen_dic(data)
    print(feature_values)
    count = 0
    for feature in data.columns[1:]:
        feature_values_list = data[feature]
        if any(x > 1.0 or x < 0.0 for x in feature_values_list):
            feature_values_list = [feature_values_list[index] / max(data[feature]) for index in range(len(feature_values_list))]
        data[feature] = feature_values_list
    print(data)
    value_total_dic = {}
    for feature in data.columns[1:]:
        value_total_dic[feature] = 0
    for t in sample.keys():
        if t[0] == 'T' and sample[t] == 1:
            value_list = [t[1:]]
            for feature in data.columns[1:]:
                value_total_dic[feature] += data[feature].iloc[int(t[1:])]
                value_list.append(feature_values[t][feature])
            count += 1
            result_df.loc[len(result_df)] = value_list

    fval_total = 0
    for feature in data.columns[1:]:
        if feature in eff_features:
            fval_total += weights[feature]*pow((value_total_dic[feature]-sum(data[feature]))/len(data), 2)
        elif feature in cost_features:
            fval_total += weights[feature]*pow(value_total_dic[feature]/len(data),2)
    fval_total += weights["num"]*pow(count/len(data),2)
    sum_df = pd.DataFrame(columns=["selected_case_num", "total_time", "total_rate", "fval", "execution_time", "max_time"])
    sum_headers = ["selected_case_num"]
    sum_list = [count]
    for feature in data.columns[1:]:
        sum_list.append(value_total_dic[feature])
        sum_headers.append("total_"+feature)
    sum_list += [fval_total, execution_time, max_time]
    sum_headers += ["fval", "execution_time", "max_time"]
    print(fval_total)

    # result_df.to_csv("BootQA1/"+data_name+"/size_"+str(sample_size)+"/result.csv")
    # sum_df.to_csv("BootQA1/" + data_name + "/size_" + str(sample_size) + "/sum.csv")


    # for t in sample.keys():
    #     if t[0] == 'T' and sample[t] == 1:
    #         total_time += foods[t]['time']
    #         total_rate += foods[t]['rate']
    #         time_item = foods[t]['time']
    #         rate_item = foods[t]['rate']
    #         time_list.append(time_item)
    #         rate_list.append(rate_item)
    #         count += 1
    #         result_df.loc[len(result_df)] = [t[1:], time_item, rate_item]
    #
    # time_list_n = [time_list[index]/max(data["time"]) for index in range(len(time_list))]
    # fval = (1/3)*pow(sum(time_list_n)/len(data),2) + (1/3)*pow((sum(rate_list)-sum(data["rate"]))/len(data),2)+(1/3)*pow(count/len(data),2)
    # sum_df.loc[len(sum_df)] = [count, total_time, total_rate, fval, execution_time, max_time]
    # result_df.to_csv("../BootQA/"+data_name+"/size_"+str(sample_size)+"/"+str(repeat)+"/result.csv")
    # sum_df.to_csv("../BootQA/" + data_name + "/size_" + str(sample_size) + "/" + str(repeat) + "/sum.csv")

def merge(sample_list):
    case_list = {}
    for i in range(len(sample_list)):
        for t in sample_list[i].keys():
            if t[0] == 'T' and sample_list[i][t] == 1:
                case_list[t] = 1
    return case_list


def sample_unique(data, m, n):
    selected_samples = []

    # Sample m times, each sample of size n
    for _ in range(m):
        sample = random.sample(data, n)
        selected_samples.extend(sample)

    # Get unique selected elements
    unique_selected = list(set(selected_samples))

    return unique_selected


if __name__ == '__main__':
    # a = random_sample_unique(100, 0.2, 5)
    # print(len(sample_unique(range(100), a, 5)))
    # sample_list = [0,1,2,3,4]
    data = get_data("paintcontrol")
    # print(len(data))
    # print(data.shape[0])
    weights = {"time": 1/3, "rate": 1/3, "num":1/3}
    subproblem_num = random_sample_unique(len(data), 0.2, 7)
    sample_list_total = bootstrap_sampling(len(data), subproblem_num, 7)
    sample_first_list, energy_first_list, execution_time, max_time = run_qpu(sample_list_total, data, 7, "paintcontrol","DEV-a0f286fa6d476ebde338230bcb8f2b66d8341f70", ["rate"],["time"],weights)
    # sample_first_list, energy_first_list, execution_time, max_time = run_qpu(sample_list_total, data, subproblem_num,7, "paintcontrol", )
    merge_sample = merge(sample_first_list)
    print_results(merge_sample, data, "paintcontrol", 7, execution_time, max_time, ["time"], ["rate"], weights)

    # print(create_bqm(sample_list, 5, 0, data, ["time"], ["rate"], weights))
    # merge_sample={"T0":0, "T10":0, "T11":1, "T20":0, "T25":0, "T26":0, "T42":0, "T54":0, "T59":0, "T82":0}
    # sample_list_total = bootstrap_sampling(len(data), 2, 5)
    # run_qpu(sample_list_total, data, 5, "paintcontrol","DEV-a0f286fa6d476ebde338230bcb8f2b66d8341f70", ["rate"],["time"],weights)
    # print_results(merge_sample, data, "paintcontrol", 1, 1, 1, ["time"], ["rate"], weights)

    # sample, data, data_name, sample_size, execution_time, max_time, eff_features, cost_features, weights
    # parser = argparse.ArgumentParser()
    # parser.add_argument('i', type=int)
    # parser.add_argument('r', type=int)
    # parser.add_argument('dn', type=str)
    # args = parser.parse_args()
    # index = args.i
    # index = int(index/10)
    # repeat = args.r
    # data_name = args.dn
    # if data_name == "gsdtsr":
    #     sample_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    #     sample_time_list = [64, 32, 21, 16, 12, 10, 9, 8, 7, 6, 5, 5, 4, 4, 4, 3]
    # elif data_name == "iofrol":
    #     sample_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    #     sample_time_list = [383, 191, 127, 95, 76, 63, 54, 47, 42, 38, 34, 31, 29, 27, 25, 23]
    # elif data_name == "paintcontrol":
    #     sample_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 89]
    #     sample_time_list = [20, 9, 6, 4, 3, 3, 2, 2, 1]
    # sample_time = sample_time_list[index]
    # sample_size = sample_size_list[index]
    # print("repeat times:"+str(repeat))
    # print("sample size",sample_size)
    # print("data name",data_name)
    # data = get_data(data_name)
    # sample_total_list = bootstrap_sampling(data, sample_time, sample_size)
    # sample_first_list, energy_first_list, execution_time, max_time = run_qpu(sample_total_list, data, sample_time, sample_size, data_name)
    # merge_sample = merge(sample_first_list)
    # print_diet(merge_sample, data, data_name, sample_size, repeat, execution_time, max_time)

