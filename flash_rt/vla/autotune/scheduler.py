"""Explicit Slurm/SSH transport, including the site's necessary local query."""

import shlex
import subprocess

from .slurm import parse_submission


class SlurmClient:
    def __init__(self, ssh_host=None, *, runner=None, timeout=60):
        self.ssh_host = ssh_host
        self.runner = runner or subprocess.run
        self.timeout = timeout

    def _run(self, argv):
        command = (['ssh', '-o', 'BatchMode=yes', self.ssh_host, shlex.join(argv)]
                   if self.ssh_host else argv)
        return self.runner(command, text=True, capture_output=True,
                           timeout=self.timeout, check=True)

    def submit(self, script):
        return parse_submission(self._run(['sbatch', '--parsable', str(script)]).stdout)

    def status(self, job_id):
        job_id = parse_submission(job_id)
        try:
            queue = self._run(['squeue', '--local', '--jobs', job_id, '--noheader',
                               '--format', '%i|%T|%R']).stdout.strip()
        except subprocess.CalledProcessError as error:
            # Slurm 24.11 single-job queries fail after controller retention
            # expires. Only that specific absence falls through to accounting;
            # transport, authorization and controller outages still propagate.
            if error.returncode != 1 or 'Invalid job id specified' not in (error.stderr or ''):
                raise
            queue = ''
        if queue:
            lines = queue.splitlines()
            if len(lines) != 1:
                raise ValueError('Expected one exact job, not an array/group query')
            observed, state, reason = lines[0].split('|', 2)
            if observed != job_id:
                raise ValueError('Scheduler returned another job')
            return {'job_id': job_id, 'state': state, 'reason': reason, 'terminal': False}
        accounting = self._run(['sacct', '-j', job_id, '-X', '--parsable2', '--noheader',
                                '-o', 'JobIDRaw,State%32,ExitCode']).stdout.strip()
        if not accounting:
            return {'job_id': job_id, 'state': 'UNKNOWN', 'terminal': False}
        rows = [line.split('|') for line in accounting.splitlines()]
        matches = [row for row in rows if row[0] == job_id]
        if len(matches) != 1 or len(matches[0]) < 3:
            raise ValueError('Accounting did not uniquely identify the requested allocation')
        _, state, exit_code = matches[0][:3]
        terminal = state.split()[0].rstrip('+') in {
            'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
        return {'job_id': job_id, 'state': state, 'exit_code': exit_code, 'terminal': terminal}
