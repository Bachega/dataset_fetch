import pandas as pd
from io import StringIO

def to_csv(text):
    data = False
    header = ""
    new_content = []
    for line in text:
        if not data:
            if "@ATTRIBUTE" in line or "@attribute" in line:
                attributes = line.split()
                if("@attribute" in line):
                    attri_case = "@attribute"
                else:
                    attri_case = "@ATTRIBUTE"
                column_name = attributes[attributes.index(attri_case) + 1]
                header = header + column_name + ","
            elif "@DATA" in line or "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                new_content.append(header)
        else:
            new_content.append(line)
    return new_content


def arff_to_csv(file):
    content = StringIO(file.getvalue().decode("utf-8")).read().splitlines()
    new = to_csv(content)
    csv_content = '\n'.join(new)
    df = pd.read_csv(StringIO(csv_content))
    return df