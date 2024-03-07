import requests
from bs4 import BeautifulSoup as BS
from urllib.parse import urljoin
import json
import sys

base_url = "https://sum20ua.com"
data_list = []
prev_omonim_lx_number = 999999

def get_url_page(page_idx):
    return f"{base_url}/?page={page_idx}&wordid=1"

def get_total_page_count(url_page):
    html = get_urls_HTML_content(url_page)
    return int(html.select("#MaxPage")[0].text.strip())


def get_urls_HTML_content(url):
    page_response = requests.get(url)
    return BS(page_response.content, "html.parser")


def find_acute_accent_positions(input_string):
    positions = []

    for idx, char in enumerate(input_string):
        if char == '́':
            positions.append(idx-1) # cuz idx - is index of acute symbol
    return positions


def remove_acute_accents(input_string):
    accent_positions = find_acute_accent_positions(input_string)

    for i, position in enumerate(accent_positions):
        position-=i
        input_string = input_string[:position + 1] + input_string[position + 2:]

    return input_string


# not used
def has_non_ukrainian_symbols(input_string):
    allowed_symbols = set("́-")  # Acute accent and hyphen are allowed

    ukrainian_letters = set("АБВГҐДЕЄЖЗИIІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиiіїйклмнопрстуфхцчшщьюя")

    for char in input_string:
        if char not in ukrainian_letters and char not in allowed_symbols:
            return True  # The string contains a non-Ukrainian symbol (except for allowed symbols)

    return False  # The string only contains Ukrainian letters, acute accent symbols, and hyphens


def remove_non_ukrainian_symbols(input_string):
    # allowed_symbols = set("́-")  # Acute accent and hyphen are allowed
    allowed_symbols = set("-")  # Acute accent and hyphen are allowed

    ukrainian_letters = set("АБВГҐДЕЄЖЗИIІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиiіїйклмнопрстуфхцчшщьюя")

    result_string = ''.join(char for char in input_string if char in ukrainian_letters or char in allowed_symbols)

    return result_string


def lemma_text_word_from_entry(block):
    return remove_acute_accents(block.find_parents("div", class_="ENTRY")[0].select(".WORD")[0].text.strip())


# INTF, INTN
def spider_sense_block(block):
    data = {
        "param": "",
        "gloss": "",
        "examples": [],
    }


    # 1) PARAM
    data["param"] = block.select(".PARAM")[0].text.strip() if len(block.select(".PARAM")) else ""

    # 2) FORMULA (GLOSS)
    gloss = block.select(".FORMULA")

    if not len(gloss):
        lemma = remove_acute_accents(block.find_parents("div", class_="ENTRY")[0].select(".WORD")[0].text.strip())
        print(f"☹️ No gloss for: {lemma}")
        return False

    data["gloss"] = gloss[0].text.strip()

    # 3) ILLs (USAGE EXAMPLES)
    usage_examples = block.select(".ILL")

    if not len(usage_examples):
        lemma = lemma_text_word_from_entry(block)
        print(f"☹️ No usage examples for: {lemma}")
        # return False  # ????

    # ---> iterate them:
    for ex in usage_examples:
        ILLTXT = ex.select(".ILLTXT")
        ILLSRC = ex.select(".ILLSRC")
        text = ILLTXT[0].text.strip() if len(ILLTXT) else ""
        src = ILLSRC[0].text.strip() if len(ILLSRC) else ""
        if text:
            data["examples"].append({
                "text": text,
                "src": src
            })

    return data

def spiderLemmaPage(lemma_url):
    # Visit the individual page of current Lemma
    html = get_urls_HTML_content(lemma_url)

    linkentry_text_content = None

    # same 🤜 🤛 :
    # int_f_and_n = html.select(".ENTRY > .INTF, .ENTRY > .INTN")
    entry = html.select("div.ENTRY:not(.LINKENTRY > .ENTRY)")[0]
    int_f_and_n = entry.select(".INTF, .INTN")


    # int_f_and_n = html.find_all('div', class_=['INTF', 'INTN'])

    if not len(int_f_and_n):
        print(f"🚫 There is no INTF', 'INTN' for lemma: {remove_acute_accents(html.select('.WORD')[0].text.strip())}")

        # 🛑 INTF,INTN can be also in LINKENTRY !!!
        int_f_and_n = html.select(".LINKENTRY .INTF, .LINKENTRY .INTN")

        if not len(int_f_and_n):
            print(f"🚫 LINKENTRY 🔗 : There is no 'INTF', 'INTN' for lemma: {remove_acute_accents(html.select('.WORD')[0].text.strip())}")
        else:
            linkentry_text_content = html.select('.WORD')[1] if html.select('.WORD')[1] else None
            if linkentry_text_content:
                linkentry_text_content = linkentry_text_content.text.strip()
            print(f'But there is for LINKENTRY 🔗: {remove_acute_accents(linkentry_text_content)}')



    # IDK if I'll stumble upon it ..
    blk_f_and_n = html.find_all('div', class_=['BLKF', 'BLKN'])
    if len(blk_f_and_n):
        raise Exception(f"🙀 There is BLKF or BLKN in ENTRY {lemma_url}")

    sense_blocks = []
    for sense_block in int_f_and_n:
        sense_block_data = spider_sense_block(sense_block)
        if not sense_block_data:
            continue
        sense_blocks.append(sense_block_data)

    return sense_blocks, linkentry_text_content

def spiderWListBlock(url_page):
    global prev_omonim_lx_number

    def checkAnchorContents(contents):
        global prev_omonim_lx_number

        if len(contents) == 1:
            prev_omonim_lx_number = 999999
            return False
        if anchor_contents[1]['class'][0].strip() != "LX":
            prev_omonim_lx_number = 999999
            return False

        new_omonim_lx_number = int(anchor_contents[1].text.strip())

        if new_omonim_lx_number == 1:
            prev_omonim_lx_number = 1
            return True

        if new_omonim_lx_number > prev_omonim_lx_number:
            prev_omonim_lx_number = new_omonim_lx_number
            return True # this lemma is a subset of omonims with prev Lemma (lemma_li)

    html = get_urls_HTML_content(url_page)
    list_block = html.select("#WListBlock > ul li")

    for lemma_li in list_block:
        lemma_anchor = lemma_li.select('a')[0]
        anchor_contents = lemma_anchor.contents

        if len(anchor_contents) == 0:
            raise Exception("Lemma's <a> has no content!")

        text_content = anchor_contents[0].text.strip()
        cleared_text_content = remove_non_ukrainian_symbols(text_content)

        if len(cleared_text_content) <= 3:
            print(f"🪫 Lemma \"{text_content}\" is too SHORT")
            continue

        # if cleared_text_content == 'АВСТРАЛІЄЦЬ':
        #     print("!!!!!!!")

        is_omomim = checkAnchorContents(anchor_contents)
        # print(isOmomim, prev_omonim_lx_number)

        # Get URL of current Lemma page
        lemma_href = lemma_anchor.get('href')

        full_lemma_url = urljoin(base_url, lemma_href)
        # gloss, usage_example = spiderLemmaPage(full_lemma_url)
        sense_blocks, linkentry_text_content = spiderLemmaPage(full_lemma_url)

        if linkentry_text_content:
            text_content = linkentry_text_content
            cleared_text_content = remove_non_ukrainian_symbols(linkentry_text_content)

        use_prev_lemma = is_omomim and prev_omonim_lx_number > 1

        if use_prev_lemma:
            data_item = data_list[-1]
        else:
            data_item = {
                "lemma": None,
                "senses": [],
                "accent_positions": []
            }

        if not use_prev_lemma and len(sense_blocks) <= 1 and prev_omonim_lx_number != 1:
            continue

        accent_positions = find_acute_accent_positions(text_content)
        if len(accent_positions) != 0:
            data_item["accent_positions"] = accent_positions

        data_item["lemma"] = cleared_text_content
        data_item["senses"] += sense_blocks

        if not use_prev_lemma:
            data_list.append(data_item)


def main():
    totalPagesCount = get_total_page_count(get_url_page(0))
    print(f"TOTAL PAGES: {totalPagesCount}")

    for page_idx in range(totalPagesCount):
        url_page = get_url_page(page_idx)  # https://sum20ua.com/?page=1&wordid=1

        spiderWListBlock(url_page)

        if (page_idx % 10 == 0 or page_idx == totalPagesCount-1) and page_idx != 0:
            # Save data to a JSON file
            output_file = f'data/parser_results/output{page_idx}.json'
            with open(output_file, 'w', encoding='utf-8') as json_file:
                json.dump(data_list, json_file, ensure_ascii=False, indent=2)

            print(f'♻️ Data saved to {output_file}.')


main()