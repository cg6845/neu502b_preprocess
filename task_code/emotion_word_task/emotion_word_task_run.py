import uuid
from pathlib import Path
import csv
import argparse

import pandas as pd
from psychopy import core, visual, event, logging
from psychopy.hardware import keyboard
from psychopy.preferences import prefs

PATH_TRIALS = 'emotion_word_task/shuffled_trials.csv'

# prefs.general['shutdownKey'] = 'escape'

def main(subid: str) -> None:
    # -- CONFIG + LOGGING -- #
    global_clock = core.Clock() #reference time 
    logging.setDefaultClock(global_clock)

    log_path = Path(f'{subid}.log')
    logging.LogFile(str(log_path), level=logging.INFO, filemode='w')
    logging.info(f'Starting session subid={subid}')

    # # set global quit key
    # event.globalKeys.clear()
    # event.globalKeys.add(key='escape', func=core.quit)

    # -- WINDOW -- #
    # win = visual.Window(size=[1000, 1000], fullscr=False, color='white', name='Window')
    win = visual.Window(fullscr=True, color='white', name='Window', units='pix')

    # -- STIMULI + DURATIONS -- #
    waiting_screen = visual.TextStim(win = win, pos = (0, 0), text = "waiting for scanner input", 
                                     color = "black", height = 60)

    fixation = visual.TextStim(win=win, pos=(0, 0), text='+', color='black', height = 60)
    fix_dur_block = 10.0

    emotion_word = visual.TextStim(win=win, pos=(0, 0), color = 'black', text='', height = 60)
    word_duration = 2.0

    iti_duration = 2.0

    # -- TRIALS -- #
    trials = pd.read_csv(PATH_TRIALS)

    # -- CRASH-SAFE CSV WRITER -- #
    out_csv_path = Path(f'{subid}.csv')
    f = out_csv_path.open('w', newline='')
    fieldnames = ['subid', 'trial', 'word']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()

    # non-slip timing add on 
    global_clock = core.Clock() #reference time 

    # waiting screen until trigger key is pressed
    waiting_screen.draw()
    win.flip()

    # wait for trigger key to start
    keys = event.waitKeys(keyList=['equal'])
    logging.info(f'Start key pressed: {keys}')

    # reset global time -- non-slip timing
    global_clock.reset() #set the current time to 0 sec
    t_next = global_clock.getTime()

    # start fixation 
    fixation.draw()
    win.flip() 

    def wait_until(target_time: float) -> None: # converts absolute time to how long to wait right now
        remaining = target_time - global_clock.getTime()
        if remaining > 0:
            core.wait(remaining)

    def show_for(stim, dur_s: float) -> None: # Use the t_next variable from the main scope
        nonlocal t_next
        stim.draw()
        win.flip()
        t_next += dur_s
        wait_until(t_next)

    try:
        for t_idx, trial in enumerate(trials.itertuples(index=False)):
            word = str(trial.word)
            logging.data(f'Trial {t_idx} start: word={word}')

            keys = event.getKeys()
            if keys:
                if 'escape' in keys:
                    win.close() # Optional: explicitly close the window first
                    core.quit()

            # Fixation block at beginning
            if t_idx == 0:
                logging.data(f"Long fixation")
                show_for(fixation, fix_dur_block)

            # Word presentation
            emotion_word.text = word
            show_for(emotion_word, word_duration)

            writer.writerow({'subid': subid, 'trial': t_idx, 'word': word})
            f.flush()

            # ITI (fixation)
            show_for(fixation, iti_duration)

            # Fixation block every 24 trials, including at end
            if (t_idx + 1) % 24 == 0:
                logging.data(f"Long fixation")
                show_for(fixation, fix_dur_block)

        logging.info('Session finished normally.')

    except Exception as e:
        logging.error(f'Session crashed: {repr(e)}')
        raise

    finally:
        try:
            f.close()
        finally:
            win.close()
            core.quit()

if __name__ == '__main__':
    subject_id = str(uuid.uuid4())
    main(subid=subject_id)
