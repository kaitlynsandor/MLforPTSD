import pandas as pd

def convert_sav_to_csv(input_file):
    df = pd.read_spss(input_file + ".sav")
    df.to_csv(input_file + ".csv", index=False)

def remove_unnecessary_columns(input_file):
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    data = pd.read_csv(input_file)
    columns = find_unnecessary_columns(input_file)
    for column in columns:
        data.pop(column)

    # drop the row if any of the values are null
    # data_new = data.dropna(thresh=data.shape[1], axis=0)
    data = data.dropna()
    return data

def find_unnecessary_columns(input_file):
    data = pd.read_csv(input_file)
    all_cols = data.columns.tolist()
    necessary_columns = all_cols[:]
    for string in necessary_columns[:]:
        string_arr = list(string)
        if string_arr[0] != 'A' and string_arr[0] != 'D':
            necessary_columns.remove(string)
        elif len(string_arr) > 2 and any(not char.isdigit() for char in string_arr[1:]):
            necessary_columns.remove(string)
        elif string_arr[0] == 'D' and ((int(string[1:]) > 27 or int(string[1:]) < 8) and int(string[1:]) != 60):
            necessary_columns.remove(string)
        # elif string_arr[0] == "A" and ((int(string[1:]) > 62 or int(string[1:]) < 27) and int(string[1:]) != 99 and not (int(string[1:]) < 106 and int(string[1:]) > 91)):
        #     necessary_columns.remove(string)
        elif string_arr[0] == "A" and ((int(string[1:]) > 46 or int(string[1:]) < 27) and int(string[1:]) != 99):
            necessary_columns.remove(string)
    return list(set(all_cols) - set(necessary_columns))

def get_admissions_to_discharge_symptoms_dict():
    startA = 27
    startD = 8
    output = dict()

    while startA < 47:
        output[str('A' + str(startA))] = str('D' + str(startD))
        startD += 1
        startA += 1
    return output

def calculate_PTSD_scores_add_to_and_return_new_dataframe(df):
    map_admissions_cols = get_admissions_to_discharge_symptoms_dict()
    admissions_scores = []
    discharge_scores = []
    responds = []
    diff_scores = []
    healeds = []
    inputs_omitteds = []
    SHI_positivity = []

    for index, row in df.iterrows():
        admissions_score = 0  # the admissions and dischrage score only take into account scores for which there is both an
                              # admissions value and a discharge calue
        discharge_score = 0
        respond = -1          # if any inprovement in admissions score to discharge score is seen
        diff_score = 0        # what the exact difference from admissions score to discharge score is
        healed = -1           # if the patient's score is now below the threshold for PTSD
        inputs_omitted = 0   # number of symptoms there is no data for
        for key in map_admissions_cols.keys():
            if row[map_admissions_cols[key]] and row[key]:
                admissions_score += row[key]
                discharge_score += row[map_admissions_cols[key]]
            else:
                inputs_omitted += 1

        if row['D60'] > 1:
            SHI_positivity.append(1)
        else:
            SHI_positivity.append(0)

        if admissions_score - discharge_score > 0:
            respond = 1
        else:
            respond = 0

        diff_score = admissions_score - discharge_score

        if discharge_score < 33:
            healed = 1
        else:
            healed = 0

        admissions_scores.append(admissions_score)
        discharge_scores.append(discharge_score)
        responds.append(respond)
        diff_scores.append(diff_score)
        healeds.append(healed)
        inputs_omitteds.append(inputs_omitted)

    df['admissions_score'] = admissions_scores
    df['discharge_score'] = discharge_scores
    df['respond'] = responds
    df['healed'] = healeds
    df['diff_score'] = diff_scores
    df['inputs_omitted'] = inputs_omitteds
    df['SHI_positivity'] = SHI_positivity
    return df