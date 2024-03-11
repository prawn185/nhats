import sqlite3

from tqdm import tqdm

# Connect to the SQLite database
conn = sqlite3.connect('nhats.db')
c = conn.cursor()

# Define the table names
table_names = ["NHATS_Round_1_SP_File",
               "NHATS_Round_2_SP_File",
               "NHATS_Round_3_SP_File",
               "NHATS_Round_4_SP_File",
               "NHATS_Round_5_SP_File",
               "NHATS_Round_6_SP_File",
               "NHATS_Round_7_SP_File",
               "NHATS_Round_8_SP_File",
               "NHATS_Round_9_SP_File",
               "NHATS_Round_10_SP_File",
               "NHATS_Round_11_SP_File",
               "NHATS_Round_12_SP_File"
               ]


def column_exists(table_name, column_name):
    c.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in c.fetchall()]
    return column_name in columns

for table_name in tqdm(table_names, desc="Processing tables"):
    #regex number from table
    round_number = table_name.split('_')[2]
    # Create the demclas column if it doesn't exist
    if not column_exists(table_name, 'demclas'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN demclas INTEGER")
    c.execute(f"UPDATE {table_name} SET demclas = -9 WHERE r{round_number}dresid = 3")
    c.execute(f"UPDATE {table_name} SET demclas = -1 WHERE r{round_number}dresid = 4")
    c.execute(f"UPDATE {table_name} SET demclas = 1 WHERE hc{round_number}disescn9 = 1 AND (is{round_number}resptype = 1 OR is{round_number}resptype = 2)")

    # Code AD8 items and score
    for num in tqdm(range(1, 9), desc="AD8 items"):
        if not column_exists(table_name, f"ad8_{num}"):
            c.execute(f"ALTER TABLE {table_name} ADD COLUMN ad8_{num} INTEGER")
        c.execute(f"UPDATE {table_name} SET ad8_{num} = -1")
        c.execute(f"UPDATE {table_name} SET ad8_{num} = NULL WHERE is{round_number}resptype = 2 AND demclas IS NULL")
        c.execute(f"UPDATE {table_name} SET ad8_{num} = 1 WHERE is{round_number}resptype = 2 AND demclas IS NULL AND (cp{round_number}chgthink{num} = 1 OR cp{round_number}chgthink{num} = 3)")
        c.execute(f"UPDATE {table_name} SET ad8_{num} = 0 WHERE is{round_number}resptype = 2 AND demclas IS NULL AND cp{round_number}chgthink{num} = 2 AND ad8_{num} IS NULL")

        if not column_exists(table_name, f"ad8miss_{num}"):
            c.execute(f"ALTER TABLE {table_name} ADD COLUMN ad8miss_{num} INTEGER")
        c.execute(f"UPDATE {table_name} SET ad8miss_{num} = -1")
        c.execute(f"UPDATE {table_name} SET ad8miss_{num} = 0 WHERE is{round_number}resptype = 2 AND demclas IS NULL AND (ad8_{num} = 0 OR ad8_{num} = 1)")
        c.execute(f"UPDATE {table_name} SET ad8miss_{num} = 1 WHERE is{round_number}resptype = 2 AND demclas IS NULL AND ad8_{num} IS NULL")
        c.execute(f"UPDATE {table_name} SET ad8_{num} = 0 WHERE is{round_number}resptype = 2 AND demclas IS NULL AND ad8_{num} IS NULL")

    # Count AD8 items and missing items
    if not column_exists(table_name, 'ad8_score'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN ad8_score INTEGER")
    c.execute(f"UPDATE {table_name} SET ad8_score = -1")
    c.execute(f"UPDATE {table_name} SET ad8_score = (ad8_1 + ad8_2 + ad8_3 + ad8_4 + ad8_5 + ad8_6 + ad8_7 + ad8_8) WHERE is{round_number}resptype = 2 AND demclas IS NULL")

    if not column_exists(table_name, 'ad8_miss'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN ad8_miss INTEGER")
    c.execute(f"UPDATE {table_name} SET ad8_miss = -1")
    c.execute(f"UPDATE {table_name} SET ad8_miss = (ad8miss_1 + ad8miss_2 + ad8miss_3 + ad8miss_4 + ad8miss_5 + ad8miss_6 + ad8miss_7 + ad8miss_8) WHERE is{round_number}resptype = 2 AND demclas IS NULL")

    # Code AD8 dementia classification
    if not column_exists(table_name, 'ad8_dem'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN ad8_dem INTEGER")
    c.execute(f"UPDATE {table_name} SET ad8_dem = 1 WHERE ad8_score >= 2")
    c.execute(f"UPDATE {table_name} SET ad8_dem = 2 WHERE ad8_score = 0 OR ad8_score = 1")

    # Update dementia classification variable with AD8 class
    c.execute(f"UPDATE {table_name} SET demclas = 1 WHERE ad8_dem = 1 AND demclas IS NULL")
    c.execute(f"UPDATE {table_name} SET demclas = 3 WHERE ad8_dem = 2 AND cg{round_number}speaktosp = 2 AND demclas IS NULL")

    # Code date items and count
    for num in range(1, 5):
        if not column_exists(table_name, f"date_item{num}"):
            c.execute(f"ALTER TABLE {table_name} ADD COLUMN date_item{num} INTEGER")
        c.execute(f"UPDATE {table_name} SET date_item{num} = cg{round_number}todaydat{num} WHERE cg{round_number}todaydat{num} > 0")
        c.execute(f"UPDATE {table_name} SET date_item{num} = 0 WHERE cg{round_number}todaydat{num} = 2 OR cg{round_number}todaydat{num} = -7")

    if not column_exists(table_name, 'date_sum'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN date_sum INTEGER")
    c.execute(f"UPDATE {table_name} SET date_sum = date_item1 + date_item2 + date_item3 + date_item4")
    c.execute(f"UPDATE {table_name} SET date_sum = -2 WHERE date_sum IS NULL AND cg{round_number}speaktosp = 2")
    c.execute(f"UPDATE {table_name} SET date_sum = -3 WHERE (date_item1 IS NULL OR date_item2 IS NULL OR date_item3 IS NULL OR date_item4 IS NULL) AND cg{round_number}speaktosp = 1")

    if not column_exists(table_name, 'date_sumr'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN date_sumr INTEGER")
    c.execute(f"UPDATE {table_name} SET date_sumr = date_sum")
    c.execute(f"UPDATE {table_name} SET date_sumr = NULL WHERE date_sum = -2")
    c.execute(f"UPDATE {table_name} SET date_sumr = 0 WHERE date_sum = -3")

    # President and Vice President name items and count
    if not column_exists(table_name, 'preslast'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN preslast INTEGER")
    c.execute(f"UPDATE {table_name} SET preslast = cg{round_number}presidna1 WHERE cg{round_number}presidna1 > 0")
    c.execute(f"UPDATE {table_name} SET preslast = 0 WHERE cg{round_number}presidna1 = -7 OR cg{round_number}presidna1 = 2")

    if not column_exists(table_name, 'presfirst'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN presfirst INTEGER")
    c.execute(f"UPDATE {table_name} SET presfirst = cg{round_number}presidna3 WHERE cg{round_number}presidna3 > 0")
    c.execute(f"UPDATE {table_name} SET presfirst = 0 WHERE cg{round_number}presidna3 = -7 OR cg{round_number}presidna3 = 2")

    if not column_exists(table_name, 'vplast'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN vplast INTEGER")
    c.execute(f"UPDATE {table_name} SET vplast = cg{round_number}vpname1 WHERE cg{round_number}vpname1 > 0")
    c.execute(f"UPDATE {table_name} SET vplast = 0 WHERE cg{round_number}vpname1 = -7 OR cg{round_number}vpname1 = 2")

    if not column_exists(table_name, 'vpfirst'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN vpfirst INTEGER")
    c.execute(f"UPDATE {table_name} SET vpfirst = cg{round_number}vpname3 WHERE cg{round_number}vpname3 > 0")
    c.execute(f"UPDATE {table_name} SET vpfirst = 0 WHERE cg{round_number}vpname3 = -7 OR cg{round_number}vpname3 = 2")

    if not column_exists(table_name, 'presvp'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN presvp INTEGER")
    c.execute(f"UPDATE {table_name} SET presvp = preslast + presfirst + vplast + vpfirst")
    c.execute(f"UPDATE {table_name} SET presvp = -2 WHERE presvp IS NULL AND cg{round_number}speaktosp = 2")
    c.execute(f"UPDATE {table_name} SET presvp = -3 WHERE presvp IS NULL AND cg{round_number}speaktosp = 1 AND (preslast IS NULL OR presfirst IS NULL OR vplast IS NULL OR vpfirst IS NULL)")

    if not column_exists(table_name, 'presvpr'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN presvpr INTEGER")
    c.execute(f"UPDATE {table_name} SET presvpr = presvp")
    c.execute(f"UPDATE {table_name} SET presvpr = NULL WHERE presvp = -2")
    c.execute(f"UPDATE {table_name} SET presvpr = 0 WHERE presvp = -3")

    # Orientation domain: Sum of date recall and president/VP naming
    if not column_exists(table_name, 'date_prvp'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN date_prvp INTEGER")
    c.execute(f"UPDATE {table_name} SET date_prvp = date_sumr + presvpr")

    # Executive function domain: Clock drawing score
    if not column_exists(table_name, 'clock_scorer'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN clock_scorer INTEGER")
    c.execute(f"UPDATE {table_name} SET clock_scorer = cg{round_number}dclkdraw")
    c.execute(f"UPDATE {table_name} SET clock_scorer = NULL WHERE cg{round_number}dclkdraw = -2 OR cg{round_number}dclkdraw = -9")
    c.execute(f"UPDATE {table_name} SET clock_scorer = 0 WHERE cg{round_number}dclkdraw = -3 OR cg{round_number}dclkdraw = -4 OR cg{round_number}dclkdraw = -7")
    c.execute(f"UPDATE {table_name} SET clock_scorer = 2 WHERE cg{round_number}dclkdraw = -9 AND cg{round_number}speaktosp = 1")
    c.execute(f"UPDATE {table_name} SET clock_scorer = 3 WHERE cg{round_number}dclkdraw = -9 AND cg{round_number}speaktosp = -1")

    # Memory domain: Immediate and delayed word recall
    if not column_exists(table_name, 'irecall'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN irecall INTEGER")
    c.execute(f"UPDATE {table_name} SET irecall = cg{round_number}dwrdimmrc")
    c.execute(f"UPDATE {table_name} SET irecall = NULL WHERE cg{round_number}dwrdimmrc = -2 OR cg{round_number}dwrdimmrc = -1")
    c.execute(f"UPDATE {table_name} SET irecall = 0 WHERE cg{round_number}dwrdimmrc = -7 OR cg{round_number}dwrdimmrc = -3")

    if not column_exists(table_name, 'drecall'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN drecall INTEGER")
    c.execute(f"UPDATE {table_name} SET drecall = cg{round_number}dwrddlyrc")
    c.execute(f"UPDATE {table_name} SET drecall = NULL WHERE cg{round_number}dwrddlyrc = -2 OR cg{round_number}dwrddlyrc = -1")
    c.execute(f"UPDATE {table_name} SET drecall = 0 WHERE cg{round_number}dwrddlyrc = -7 OR cg{round_number}dwrddlyrc = -3")

    if not column_exists(table_name, 'wordrecall0_20'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN wordrecall0_20 INTEGER")
    c.execute(f"UPDATE {table_name} SET wordrecall0_20 = irecall + drecall")

    # Create cognitive domains for all eligible
    if not column_exists(table_name, 'clock65'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN clock65 INTEGER")
    c.execute(f"UPDATE {table_name} SET clock65 = 0 WHERE clock_scorer > 1 AND clock_scorer <= 5")
    c.execute(f"UPDATE {table_name} SET clock65 = 1 WHERE clock_scorer >= 0 AND clock_scorer <= 1")

    if not column_exists(table_name, 'word65'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN word65 INTEGER")
    c.execute(f"UPDATE {table_name} SET word65 = 0 WHERE wordrecall0_20 > 3 AND wordrecall0_20 <= 20")
    c.execute(f"UPDATE {table_name} SET word65 = 1 WHERE wordrecall0_20 >= 0 AND wordrecall0_20 <= 3")

    if not column_exists(table_name, 'datena65'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN datena65 INTEGER")
    c.execute(f"UPDATE {table_name} SET datena65 = 0 WHERE date_prvp > 3 AND date_prvp <= 8")
    c.execute(f"UPDATE {table_name} SET datena65 = 1 WHERE date_prvp >= 0 AND date_prvp <= 3")

    # Create cognitive domain score
    if not column_exists(table_name, 'domain65'):
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN domain65 INTEGER")
    c.execute(f"UPDATE {table_name} SET domain65 = clock65 + word65 + datena65")

    # Update cognitive classification
    c.execute(f"UPDATE {table_name} SET demclas = 1 WHERE demclas IS NULL AND (cg{round_number}speaktosp = 1 OR cg{round_number}speaktosp = -1) AND (domain65 = 2 OR domain65 = 3)")
    c.execute(f"UPDATE {table_name} SET demclas = 2 WHERE demclas IS NULL AND (cg{round_number}speaktosp =1 OR cg{round_number}speaktosp = -1) AND domain65 = 1")
    c.execute(f"UPDATE {table_name} SET demclas = 3 WHERE demclas IS NULL AND (cg{round_number}speaktosp = 1 OR cg{round_number}speaktosp = -1) AND domain65 = 0")
# Commit the changes and close the connection

conn.commit()
conn.close()

## now report on how many records were updated
# Connect to the SQLite database
conn = sqlite3.connect('nhats.db')
c = conn.cursor()
for table_name in tqdm(table_names, desc="Processing tables"):
    c.execute(f"SELECT COUNT(*) FROM {table_name} WHERE demclas IS NOT NULL")
    print(f"Table {table_name} updated {c.fetchone()[0]} records")
conn.close()