

import csv
import stats as stats_data
import pandas as pd
import helpers as helper
import json
import ast


lcopf_file_path = stats_data.lcopf_forms_file_path
# lemmas_file_path = stats_data.merged_file_path
lemmas_file_path = "data/final_results/clustering_sum14_merged.json"


# lcopf_forms_not_found_file_path = stats_data.lcopf_forms_not_found_file_path

# lcopf_forms_found_file_path = stats_data.lcopf_forms_found_file_path
lcopf_forms_found_file_path = 'data/final_results/clustering_forms_found.jsonl'


def main():
    pd.set_option('display.max_colwidth', None)

    lemmas_data = pd.read_json(lemmas_file_path, lines=True)
    not_found_lemmas_df = pd.DataFrame(columns=lemmas_data.columns)

    lcopf_df = pd.read_csv(lcopf_file_path)

    print(lemmas_data.shape)
    print(lcopf_df.shape)
    print("f")

    # Проходимо по усіх "lemma" з lemmas_data
    for index, row in lemmas_data.iterrows():
        lemma = row['lemma']

        # if lemma == "австралійці":
        #     print("stop")
        
        # Перевіряємо чи "lemma" є в lcopf_df під ключем "lemma_no_accent"
        if lemma in lcopf_df['lemma_no_accent'].values:
            # Отримуємо індекс рядка в lcopf_df, де 'lemma_no_accent' рівна поточній "lemma"
            lcopf_index = lcopf_df.index[lcopf_df['lemma_no_accent'] == lemma][0]

            # Додаємо до "lemma" в lemmas_data поле "forms" зі значенням з поля "forms_no_accent" в lcopf_df
            lemmas_data.at[index, 'forms'] = lcopf_df.at[lcopf_index, 'forms_no_accent']
        else:
            # Якщо "lemma" не знайдено, перевіряємо синоніми
            found = False
            for synonym in row['synonyms']:
                synonym_lemma = synonym['lemma']

                if not synonym_lemma in lcopf_df['lemma_no_accent'].values:
                    continue

                synonym_forms = lcopf_df.loc[lcopf_df['lemma_no_accent'] == synonym_lemma, 'forms_no_accent']

                synonym_forms_list = synonym_forms.apply(ast.literal_eval).tolist()[0]

                # if not any(lemma in forms for forms in synonym_forms):
                #     continue

                # Знайдено синонім у lcopf_df, додаємо до "lemma" в lemmas_data поле "forms" зі значенням з поля "forms_no_accent" в lcopf_df

                # Check if 'forms' is initialized as a list
                if not isinstance(lemmas_data.at[index, 'forms'], list):
                    lemmas_data.at[index, 'forms'] = []

                lemmas_data.at[index, 'forms'].extend(synonym_forms_list)
                
                # print(lemmas_data.at[index, 'forms'])
                found = True
                # break
            if not found:
                # print(f"No forms for lemma: {lemma}")
                # not_found_lemmas_df = not_found_lemmas_df.append(row, ignore_index=True)
                not_found_lemmas_df = pd.concat([not_found_lemmas_df, row.to_frame().T], ignore_index=True)
                lemmas_data = lemmas_data.drop(index, axis=0)
            else:
                lemmas_data.at[index, 'forms'] = list(set(lemmas_data.at[index, 'forms']))
                # print(lemmas_data.at[index, 'forms'])


    print(f"not_found_lemmas_df shape: {not_found_lemmas_df.shape} ")
    print(not_found_lemmas_df.info())


    print(f"found_lemmas_df shape: {lemmas_data.shape} ")
    print(lemmas_data.info())



    # not_found_lemmas_df.to_json(lcopf_forms_not_found_file_path, orient='records', lines=True, default_handler=lambda x: str(x), force_ascii=False)

    lemmas_data.to_json(lcopf_forms_found_file_path, orient='records', lines=True, default_handler=lambda x: str(x), force_ascii=False)


if __name__ == '__main__':
    main()