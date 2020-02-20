#  Copyright (c) Hardy (hardy.oei@gmail.com) 2020.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
import drmaa


def main():
    with drmaa.Session() as s:
        jt = s.createJobTemplate()
        jt.remoteCommand = os.path.join(os.getcwd(), 'simple.sh')
        jobids = {}
        output = []
        for i in range(10):
            jt.args = [str(i)]
            jt.joinFiles = True
            jobid = s.runJob(jt)
            print(f'Your job {str(i)} has been submitted with id {jobid}')
            jobids[jobid] = i
        for jobid, i in jobids.items():
            retval = s.wait(jobid, drmaa.Session.TIMEOUT_WAIT_FOREVER)
            result = list(open(f'simple_{i}'.csv).readlines())[0]
            print(f'{retval.jobId} has finished with status {retval.hasExited}')
            output.append(result)
        s.deleteJobTemplate(jt)
        open('all_output.csv').write('\n'.join(output))

if __name__ == '__main__':
    main()