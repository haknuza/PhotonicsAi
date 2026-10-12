#!/usr/bin/env python3
"""Record a dedicated terminal-style window displaying a fresh, live agent run.

Requires tkinter and GStreamer (ximagesrc, vp8enc, webmmux) on an X11/XWayland
desktop. Captures only this window, never the entire desktop. No transcript replay.
"""
import argparse
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen2.5:3b')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new run directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    video = args.output.parent / (args.output.name + '.webm')
    root = tk.Tk()
    root.title('Homework 2 Task 2 - LIVE autonomous focusing agent')
    root.geometry('1100x720')
    pane = tk.Text(root, bg='#101820', fg='#e5edf5', font=('Monospace', 12), wrap='word')
    pane.pack(fill='both', expand=True)
    pane.insert('end', 'LIVE MODEL RUN - noisy simulated optical bench\n'
                'User request: Focus the beam; choose experiment settings autonomously.\n'
                'Waiting for model decisions. All subsequent output is from the actual process.\n\n')
    root.update()
    recorder = subprocess.Popen([
        'gst-launch-1.0', '-e', 'ximagesrc', f'xid={root.winfo_id()}',
        'use-damage=false', 'show-pointer=false', '!', 'video/x-raw,framerate=5/1',
        '!', 'videoconvert', '!', 'vp8enc', 'deadline=1', 'target-bitrate=1500000',
        '!', 'webmmux', '!', 'filesink', f'location={video.resolve()}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    events = queue.Queue()
    process = None

    def run():
        nonlocal process
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/mpl-task2')
        command = [sys.executable, '-u', str(Path(__file__).with_name('run_focus.py')),
                   '--model', args.model, '--output', str(args.output)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env)
        with video.with_suffix('.log').open('w') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                events.put(line)
        events.put(('done', process.wait()))

    def finish():
        if process and process.poll() is None:
            process.terminate()
        if recorder.poll() is None:
            recorder.send_signal(signal.SIGINT)
        try:
            recorder.wait(timeout=15)
        except subprocess.TimeoutExpired:
            recorder.terminate()
        if recorder.returncode:
            print(recorder.stderr.read().decode(), file=sys.stderr)
        root.destroy()

    def poll():
        while not events.empty():
            event = events.get_nowait()
            if isinstance(event, tuple):
                pane.insert('end', f'\nAgent exited with status {event[1]}. Recording: {video}\n')
                pane.see('end')
                root.after(3000, finish)
                return
            pane.insert('end', event)
            pane.see('end')
        root.after(100, poll)

    root.protocol('WM_DELETE_WINDOW', finish)
    root.after(1000, lambda: threading.Thread(target=run, daemon=True).start())
    root.after(100, poll)
    root.mainloop()
    if not video.exists() or video.stat().st_size < 1000:
        raise RuntimeError('Screen recording failed; inspect GStreamer diagnostics')
    print(f'Saved live screen recording: {video}')


if __name__ == '__main__':
    main()
