import arxiv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", type=str, help="query string")
parser.add_argument("-f", "--file", type=str, help="output file")
args = parser.parse_args()

result = arxiv.query(search_query=args.query, iterative=False, max_results=300)
title_list = [item['title'] for item in result]
title_lines = '\n'.join(title_list)
print(title_lines)
# write train.txt file
out_text_file = open(args.file, "w")
out_text_file.write(title_lines)
out_text_file.close()
