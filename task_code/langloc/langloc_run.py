import uuid
from pathlib import Path
import csv
import argparse


import pandas as pd
from psychopy import core, visual, event, logging

PATH_TRIALS = 'langloc/langloc_fmri_run1_stim_set1.csv'
PATH_IMAGE = 'langloc/hand-press-button.jpeg'


def main(subid: str) -> None:
    # -- CONFIG + LOGGING -- #
    # main_clock = core.Clock()
    global_clock = core.Clock() #reference time 
    logging.setDefaultClock(global_clock)

    log_path = Path(f'{subid}.log')
    logging.LogFile(str(log_path), level=logging.INFO, filemode='w')
    logging.info(f'Starting session subid={subid}')

    # -- WINDOW -- #
    # win = visual.Window(size=[800, 800], fullscr=False, color='white', name='Window')
    win = visual.Window(fullscr=True, color='white', name='Window', units='pix')

    # -- STIMULI + DURATIONS -- #
    waiting_screen = visual.TextStim(win = win, pos = (0, 0), text = "waiting for scanner input", 
                                     color = "black", height = 60)
    
    fixation = visual.TextStim(win=win, pos=(0, 0), text='+', color='black', height = 60)
    fix_dur_block = 14.0

    langloc = visual.TextStim(win=win, pos=(0, 0), color = 'black', text='', height = 60)
    word_duration = 0.45
    sentence_duration = 5.40

    # button press image
    button_press_img = visual.ImageStim(win = win, image = PATH_IMAGE)
    button_press_duration = 0.40

    iti_duration = 0.20
    iti = visual.TextStim(win=win, pos=(0, 0), text='')

    total_trial_duration = 6.0

    # -- TRIALS -- #
    trials = pd.read_csv(PATH_TRIALS)

    # -- CRASH-SAFE CSV WRITER -- #
    out_csv_path = Path(f'{subid}_langloc.csv')
    f = out_csv_path.open('w', newline='')
    fieldnames = ['subid', 'trial', 'category']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()

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
        for t_idx, trial in trials.iterrows():
            
            trial_num = t_idx + 1
            category = str(trial["stim14"])

            # print("starting logging")

            logging.data(f'Trial {trial_num} start: category={category}')
            
            # print(trial)   
            # print("starting trial")

            keys = event.getKeys()
            if keys:
                if 'escape' in keys:
                    win.close() # Optional: explicitly close the window first
                    core.quit()

            # Long fixation block at beginning of experiment 
            if t_idx == 0:
                logging.data(f"Long fixation")
                show_for(fixation, fix_dur_block)
            
            # print("after fixation")
            
            # For each word in the sentence
            for i in range(2, 14):
                # extract correct word
                # print("extracting current word")
                # print(i)
                stim_id = f"stim{i}"
                #print(stim_id)
                word = trial[stim_id]
                #print(word)

                # show word
                langloc.text = word
                show_for(langloc, word_duration)
            
            
            # write row in the log -- ADD TIME?
            writer.writerow({'subid': subid, 'trial': t_idx, 'category': category})
            f.flush()

            # button press image
            show_for(button_press_img, button_press_duration)

            # ITI (fixation)
            show_for(iti, iti_duration)

            # Long fixation block every 12 trials at the end of the trial, including at end
            if trial_num % 12 == 0:
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
