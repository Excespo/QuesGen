import csv
import json
import argparse
import os.path as osp
import os
from operator import itemgetter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=None, type=str, required=True,
                        help="The root directory of the raw WebSRC dataset; The output SQuAD-style json file will also"
                             "be placed here.")
    parser.add_argument("--version", default=None, type=str, required=True,
                        help="The version of the generating dataset, which will also be the name of the json file.")
    parser.add_argument("--suffix", default="", type=str,
                        help="Other suffix to distinguish different dataset.")
    return parser.parse_args()


def convert_csv_to_dict(args):
    dir_list = os.walk(args.root_dir)
    print('Start Converting')

    data, websites, qas, answers = [], [], [], []
    # The four items is recording:
    # qas: question, (item's) id, answers dict
    # websites: qas, page_id pair dict
    # answers: text, element_id, answer_start dict
    # data: 
    last_domain = None
    # The design for 'last' items, including last_domain and last is for
    # 
    # 

    for d, _, fs in dir_list:
        for f in fs:
            if f != 'dataset.csv':
                continue
            # the loops aim at processing the data in dir: /data/domain/no/
            # (in each dir there is a dataset.csv, including columns:
            # question, id, element_id, answer_start, answer)
            print('Now converting', d + '/' + f)
            raw_data = list(csv.DictReader(open(osp.join(d, f))))
            # print(raw_data) # list of ordered dict with key: column name and value: item
            curr_domain = d.split('/')[-2] # current file's domain: auto/movie...
            if last_domain != curr_domain and last_domain is not None:
                # in this case, the loop reaches a new domain, end the last dir's work
                domain = {'domain': last_domain, 'websites': websites}
                data.append(domain)
                websites = []
            last_domain = curr_domain # step the last domain's state

            raw_data.sort(key=itemgetter('id')) 
            # print(raw_data) # sort in 'id' ascending order

            last = raw_data[0]
            # print(f"An example: raw_data[0] is {last}, q is {last['question']}, i is {last['id']}, answers are {{}}, last['id'][:-5] is {last['id'][:-5]}")
            for i in range(len(raw_data)):
                current = raw_data[i] # get the i-th line of the dataset
                if i != 0:
                    qa = {'question': last['question'],
                          'id'      : last['id'],
                          'answers' : answers}  # , 'type': last['type']}
                    qas.append(qa)
                    answers = []
                if last['id'][:-5] != current['id'][:-5]:
                    # 2~-5 digits represent the website's page id
                    website = {'qas': qas, 'page_id': last['id'][2:-5]}
                    websites.append(website)
                    qas = []
                answer = {'text'        : current['answer'],
                          'element_id'  : int(current['element_id']),
                          'answer_start': int(current['answer_start'])}
                answers.append(answer)
                last = current

            if len(answers) > 0:
                qa = {'question': last['question'],
                      'id'      : last['id'],
                      'answers' : answers}  # , 'type'    : last['type']}
                qas.append(qa)
                answers = []
            if len(qas) > 0:
                website = {'qas': qas, 'page_id': last['id'][2:-5]}
                websites.append(website)
                qas = [] # add the last website info?

    domain = {'domain': last_domain, 'websites': websites}
    data.append(domain)
    dataset = {'version': args.version, 'data': data}
    print('Converting Finished\n')

    return dataset


def dataset_split(args, dataset):
    def count(last, curr):
        if last is None:
            return False
        if last != curr:
            return False
        return True

    split = json.load(open(osp.join(args.root_dir, 'dataset_split.json')))
    data = dataset['data']
    count_website = set()
    for domain in data:
        for website in domain['websites']:
            count_website.add(domain['domain'][0:2] + website['page_id'][0:2])
    print('The number of total websites is', len(count_website))

    train_list = []
    dev_list, test_list = split['dev'], split['test']
    for website in count_website:
        if website not in dev_list and website not in test_list:
            train_list.append(website)
    print('The train websites list is', train_list)
    print('The test websites list is', test_list)
    print('The dev websites list is', dev_list)

    train_data, test_data, dev_data = [], [], []
    cnt = 0
    for domain in data:
        train_websites, test_websites, dev_websites = [], [], []
        last = None
        for website in domain['websites']:
            if not count(last, website['page_id'][0:2]):
                last = website['page_id'][0:2]
                cnt += 1
            name = domain['domain'][0:2] + website['page_id'][0:2]
            if name in test_list:
                test_websites.append(website)
                continue
            if name in dev_list:
                dev_websites.append(website)
                continue
            if len(train_list) != 0 and name not in train_list:
                continue
            train_websites.append(website)
        if len(train_websites) != 0:
            train_data.append({'domain': domain['domain'], 'websites': train_websites})
        if len(test_websites) != 0:
            test_data.append({'domain': domain['domain'], 'websites': test_websites})
        if len(dev_websites) != 0:
            dev_data.append({'domain': domain['domain'], 'websites': dev_websites})
    print('The number of processed websites is', cnt)
    train_dataset = {'version': dataset['version'], 'data': train_data}
    with open(osp.join(args.root_dir, dataset['version'] + '_train_' + args.suffix + '.json'), 'w') as f:
        f.write(json.dumps(train_dataset))
    test_dataset = {'version': dataset['version'], 'data': test_data}
    with open(osp.join(args.root_dir, dataset['version'] + '_test_' + args.suffix + '.json'), 'w') as f:
        f.write(json.dumps(test_dataset))
    dev_dataset = {'version': dataset['version'], 'data': dev_data}
    with open(osp.join(args.root_dir, dataset['version'] + '_dev_' + args.suffix + '.json'), 'w') as f:
        f.write(json.dumps(dev_dataset))
    return


if __name__ == '__main__':
    args = parse_args()
    dataset = convert_csv_to_dict(args)
    dataset_split(args, dataset)

