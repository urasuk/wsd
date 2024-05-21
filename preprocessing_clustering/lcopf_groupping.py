import stats as stats_data
import pandas as pd
import helpers as helper

input_file_path = stats_data.lcopf_file_path
output_file_path = stats_data.lcopf_forms_file_path

def main():

    # Assuming df is your DataFrame
    # Read the CSV file
    input_df = pd.read_csv(input_file_path)

    # % додав через проблему аба́тський Аба́тський
    input_df['lemma'] = input_df['lemma'].apply(lambda x: x.lower())

    #! ще можна у input_df з леми забрати наголоси

    # Extract the 'lemma' and 'form' columns
    selected_columns = input_df[['lemma', 'form']]

    # Group by 'lemma' and aggregate 'form' into a list
    df = selected_columns.groupby('lemma')['form'].agg(list).reset_index()


    df['lemma_no_accent'] = df['lemma'].apply(lambda x: helper.remove_acute_accents(x,
                                                            helper.find_acute_accent_positions(x)).lower())

    
    df['lemma_accent_pos'] = df['lemma'].apply(lambda x: helper.find_acute_accent_positions(x))


    df['forms_no_accent'] = df['form'].apply(lambda forms: [helper.remove_acute_accents(form,
                                                            helper.find_acute_accent_positions(form)) for form in forms])

    df['forms_no_accent'] = df['forms_no_accent'].apply(lambda forms: [form.lower() for form in forms])

    # % додав через проблему аба́тський Аба́тський
    df['forms_no_accent'] = df['forms_no_accent'].apply(lambda forms: list(set(forms)))

    df = df[['lemma_no_accent', 'lemma_accent_pos', 'forms_no_accent']]

    print(df.head(10))

    df.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    main()