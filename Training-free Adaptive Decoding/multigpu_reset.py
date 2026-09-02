import glob, os
import sys
gpu_folder = sys.argv[1]


for f in glob.glob(os.path.join(gpu_folder, '*.sh')):
    runstr = "===running above==="
    script_file = open(f, "r")
    scripts = script_file.readlines()
    scripts = [script.strip() for script in scripts]
    script_file.close()

    if runstr in scripts:
        scripts.remove(runstr)
        print(f"removing {runstr} from {f}")
        with open(f, 'w') as f_w:
            for s in scripts:
                f_w.write(s + '\n')

        alternative_file = os.path.dirname(f) + '_tmp/' + os.path.basename(f)
        if os.path.exists(alternative_file):
            print("removing", alternative_file)
            os.remove(alternative_file)
