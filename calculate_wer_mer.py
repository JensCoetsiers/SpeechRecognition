from collections import defaultdict
import jiwer
import json

words = ['ik', 'hij', 'zij', 'aan', 'achter', 'bij', 'binnen', 'boven',
         "buiten", "dankzij", "door", "gedurende", "in", "langs", "naar", "nabij",
         "om", "omstreeks", "over", "per", "qua", "rond", "sinds", "te", "tegen",
         "tegenover", "tot", "tussen", "uit", "van", "vanaf", "vanuit", "via",
         "volgens", "voorbij", "wegens",
         "zonder", 'en', 'ofwel', 'maar', 'want', "terwijl", "om",
         "omdat", "doordat", "zodat", "zodra", "als", "toen", "hoewel",
         "tenzij", 'de', 'het', 'een', 'uh', 'dat', 'ja', 'nee', 'dan']


def calculate_wer(reference_path, generated_path):
    wer_dict = {}
    # Using defaultdict for error_words
    error_words = defaultdict(lambda: [0, 0])

    with open(reference_path, 'r') as ref_file, open(generated_path, 'r') as gen_file:
        ref_content = ref_file.readlines()
        gen_content = gen_file.readlines()

    reference_dict = {}
    generated_dict = {}

    for ref_line in ref_content:
        if "|" not in ref_line:
            print(
                f"Line '{ref_line}' from `ref_content` is not valid.")
            continue
        key, value = ref_line.strip().split("|")
        if key.endswith(".wav"):
            key = key[:-4]
        if key in reference_dict:
            print(
                f"Key '{key}' is already in `reference_dict` with value '{reference_dict[key]}'")
            continue
        reference_dict[key] = value

        # Count the absolute occurrences of each word in the reference
        ref_words = value.split()
        for word in ref_words:
            error_words[word][1] += 1

    for gen_line in gen_content:
        if "|" not in gen_line:
            print(
                f"Line '{gen_line}' from `gen_content` is not valid.")
            continue
        key, value = gen_line.strip().split("|")
        if key.endswith(".wav"):
            key = key[:-4]
        if key in generated_dict:
            print(
                f"Key '{key}' is already in `generated_dict` with value '{generated_dict[key]}'")
            continue
        generated_dict[key] = value

    for ref_key, ref_value in reference_dict.items():
        if ref_key not in generated_dict:
            print(f"Key '{ref_key}' not found in `generated_dict`")
            with open('./not_in_trans.txt', 'a') as f:
                f.write(f'{ref_key} \n')
            continue

        gen_value = generated_dict[ref_key]

        wer = jiwer.wer(ref_value, gen_value)
        mer = jiwer.mer(ref_value, gen_value)
        cer = jiwer.cer(ref_value, gen_value)

        if ref_key in wer_dict:
            wer_dict[ref_key].append((wer, mer, cer, ref_value, gen_value))
        else:
            wer_dict[ref_key] = [(wer, mer, cer, ref_value, gen_value)]

        # Identify the most wrongly transcribed words based on the reference
        gt_words = ref_value.split()
        hyp_words = gen_value.split()
        updated_words = set()  # To keep track of already updated words
        for gt, hyp in zip(gt_words, hyp_words):
            if gt != hyp and gt not in updated_words:
                error_words[gt][0] += 1  # Incrementing absolute count
                updated_words.add(gt)     # Marking word as updated

    wer_list = [{'filename': filename,
                 'wer': sum([item[0] for item in wer_scores]) / len(wer_scores),
                 'mer': sum([item[1] for item in wer_scores]) / len(wer_scores),
                 'cer': sum([item[2] for item in wer_scores]) / len(wer_scores),
                 'ref_line': [item[3] for item in wer_scores],
                 'gen_line': [item[4] for item in wer_scores]}
                for filename, wer_scores in wer_dict.items()]

    # Display the top 5 most wrongly transcribed words
    top_n = 10
    print(f"\nTop {top_n} most wrongly transcribed words:")
    sorted_error_words = sorted(
        error_words.items(), key=lambda x: x[1][0], reverse=True)
    for word, counts in sorted_error_words[:top_n]:
        print(f"{word}: {counts[0]} ({counts[0]}/{counts[1]})")

    return wer_list


def save_wer_results(wer_list, output_file):
    with open(output_file, 'w') as f:
        json.dump(wer_list, f)


def main():

    print("Available transcripts:")
    print("1. VariaNTS")
    print("2. CGN comp c&d")
    print("3. CGN comp a")
    print("4. Dialectloket")

    choice = input("Enter your choice (1,2,3 or 4): ")

    if choice == "1":
        ref_trans = './data/transcripts_var.txt'
    elif choice == "2":
        ref_trans = './data/transcripts_cgn_cd.txt'
    elif choice == "3":
        ref_trans = './data/transcripts_cgn_a.txt'
    elif choice == "4":
        ref_trans = './data/dialectloket_trans.txt'

    gen_trans = input('generated transcripts path: ')
    wer_scores = calculate_wer(
        ref_trans, gen_trans)

    average_wer = sum(item['wer'] for item in wer_scores) / len(wer_scores)
    average_mer = sum(item['mer'] for item in wer_scores) / len(wer_scores)
    average_cer = sum(item['cer'] for item in wer_scores) / len(wer_scores)
    print("\nAverage WER:", average_wer)
    print(f"Average WAR: {100*(1-average_wer)}%")
    print("Average MER:", average_mer)
    print(f"Average CER: {average_cer}%")

    save_loc = input('Where do you want to save it: ')
    save_wer_results(wer_scores, save_loc)


if __name__ == "__main__":
    main()
