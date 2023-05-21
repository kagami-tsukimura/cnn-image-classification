import subprocess


def mkdir(OUTPUT, dir):
    """Create directory.
    Args:
        dir: directory path
    """

    cmd = f"mkdir -p {OUTPUT}/{dir}"
    subprocess.call(cmd.split())


def mv_file(img, OUTPUT, dir):
    """Move file.
    Args:
        img: image path
        dir: directory path
    """

    cmd = f"mv {img} {OUTPUT}/{dir}"
    subprocess.call(cmd.split())
