import re

IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")
LINK_PATTERN = re.compile(r"\[(.*?)\]\(.*?\)")
MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")
MULTINEWLINE_PATTERN = re.compile(r"\n{3,}")

def clean_markdown(text: str) -> str:
    text = IMAGE_PATTERN.sub("", text)
    text = LINK_PATTERN.sub(r"\1", text)

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[▢■▪●]+", "", text)

    text = MULTISPACE_PATTERN.sub(" ", text)
    text = MULTINEWLINE_PATTERN.sub("\n\n", text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line in {"---", "***", "___"}:
            continue
        lines.append(line)

    return "\n".join(lines).strip()
