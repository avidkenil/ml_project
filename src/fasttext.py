def write_fasttext_format(x_df, y_df, columns, output_base_path_stub):
    out_data = {
        column: []
        for column in columns
    }

    for (i, x_row), (_, y_row) in \
            zip(x_df.iterrows(), y_df.iterrows()):
        for label in out_data:
            line = ""
            if y_row[label]:
                line += f"__label__{label} {x_row['comment_text']}"
            else:
                line += f"__label__None {x_row['comment_text']}"
            out_data[label].append(line)
    for label, txt in out_data.items():
        with open(f"{output_base_path_stub}_{label}.txt", "w") as f:
            f.write("\n".join(txt))
