import subprocess

commands = ["python chorin_generator.py",
            "python chorin_generator.py --flags 'lu'",
            "python lid_generator.py",
            "python lid_generator.py --flags 'lu'"]

for command in commands:
    print('Now running')
    print(command)
    subprocess.run(command,
                   shell=True,
                   stdout=subprocess.PIPE)
    print('Done')
