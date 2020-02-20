import random
import csv
import os

base_path = os.getcwd()
data_path = base_path + '\\data_samples\\'
master_data_list = []


# functions to create data objects
def create_wyvern():
    max_count = 478
    x = 0
    wings = [2, 4]

    while x <= max_count:
        wyvern = {}
        wyvern['Legs'] = 2
        wyvern['Wings'] = random.choice(wings)
        wyvern['Length'] = random.randrange(115, 150)
        wyvern['Height'] = random.randrange(175, 225)
        wyvern['Environment'] = 'volcanoes'
        wyvern['Classification'] = 'Wyvern'

        master_data_list.append(wyvern)
        x += 1


def create_serpent():
    max_count = 572
    x = 0
    legs = [0, 4]
    environment = ['air', 'water', 'woods', 'swamp']

    while x <= max_count:
        serpent = {}
        serpent['Legs'] = random.choice(legs)
        serpent['Wings'] = 0
        serpent['Length'] = random.randrange(215, 250)
        serpent['Height'] = random.randrange(50, 80)
        serpent['Environment'] = random.choice(environment)
        serpent['Classification'] = 'Serpent'

        master_data_list.append(serpent)
        x += 1


def create_flying_serpent():
    max_count = 548
    x = 0
    wings = [2, 4]

    while x <= max_count:
        flying_serpent = {}
        flying_serpent['Legs'] = 0
        flying_serpent['Wings'] = random.choice(wings)
        flying_serpent['Length'] = random.randrange(250, 315)
        flying_serpent['Height'] = random.randrange(99, 125)
        flying_serpent['Environment'] = 'air'
        flying_serpent['Classification'] = 'Flying Serpent'

        master_data_list.append(flying_serpent)
        x += 1


def create_dragon():
    max_count = 478
    x = 0
    wings = [2, 4, 6]
    environment = ['mountain', 'volcanoes', 'cave']

    while x <= max_count:
        dragon = {}
        dragon['Legs'] = 4
        dragon['Wings'] = random.choice(wings)
        dragon['Length'] = random.randrange(125, 175)
        dragon['Height'] = random.randrange(65, 100)
        dragon['Environment'] = random.choice(environment)
        dragon['Classification'] = 'Dragon'

        master_data_list.append(dragon)
        x += 1


def create_drake():
    max_count = 472
    x = 0
    legs = [2, 4]
    environment = ['desert', 'volcanoes', 'water']

    while x <= max_count:
        drake = {}
        drake['Legs'] = random.choice(legs)
        drake['Wings'] = 2
        drake['Length'] = random.randrange(185, 225)
        drake['Height'] = random.randrange(50, 75)
        drake['Environment'] = random.choice(environment)
        drake['Classification'] = 'Drake'

        master_data_list.append(drake)
        x += 1


def create_wyrm():
    max_count = 556
    x = 0
    environment = ['cave', 'swamp']

    while x <= max_count:
        wyrm = {}
        wyrm['Legs'] = 4
        wyrm['Wings'] = 0
        wyrm['Length'] = random.randrange(295, 345)
        wyrm['Height'] = random.randrange(35, 50)
        wyrm['Environment'] = random.choice(environment)
        wyrm['Classification'] = 'Wyrm'

        master_data_list.append(wyrm)
        x += 1


# function to assign numbers to each creature
def add_numbers(some_list):
    creature_count = 0
    for item in some_list:
        creature_count += 1
        item['Number'] = creature_count


# function to save data to a csv
def create_csv(some_file, some_list):
    with open(data_path + some_file, 'w+', newline='') as f:
        writer = csv.DictWriter(f, ['Number', 'Legs', 'Wings', 'Length', 'Height', 'Environment', 'Classification'])
        writer.writeheader()

        for item in some_list:
            writer.writerow(item)


# calling all functions
create_wyvern()
create_serpent()
create_flying_serpent()
create_dragon()
create_drake()
create_wyrm()

# shuffle the data and split it into a training and test set
random.shuffle(master_data_list)
test_list = master_data_list[-50:]
del master_data_list[-50:]

add_numbers(master_data_list)
add_numbers(test_list)

# store the data in spreadsheets
create_csv('creatures_data_set.csv', master_data_list)
create_csv('creatures_test_set.csv', test_list)

