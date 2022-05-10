import PySimpleGUI as sg
import os.path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import data_import
from evaluation import evaluate_single_patient
from operator import itemgetter


sg.theme('GreenTan')


menu_def = [['File', ['Open', 'Save', 'Exit']],
            ['Help', ['Support'], ],
            ['About', 'About...'], ]


# ---------------------- GLOBAL VARIABLES -------------------------#

data_file=[]

#Separated ERG recordings:
#Left eye:
left_dark_001_time = []
left_dark_001_data = []
left_dark_3_time = []
left_dark_3_data = []
left_light_3_time = []
left_light_3_data =[]
left_light_30_time = []
left_light_30_data = []

#Right eye:
right_dark_001_time= []
right_dark_001_data = []
right_dark_3_time= []
right_dark_3_data =[]
right_light_3_time= []
right_light_3_data = []
right_light_30_time= []
right_light_30_data = []


#Output header:
patient_text_output = []


path_to_mydata=''

# ---------------------- FUNCTIONS -------------------------#

def sep(time, uv):
    new_w_time=[]
    new_w_data=[]
    time = np.delete(time, 0)
    uv = np.delete(uv, 0)
    for k in range(len(time)):
        if time[k] == 'nan' or time[k] == 'ms':
            break
        if uv[k] == 'nan' or uv[k] == 'uV':
            break
        new_w_time.append(float(time[k]))
        new_w_data.append(float(uv[k]))

    return new_w_time, new_w_data


def recordings_seperator(data_file, tested_eye, stim_freq):
    time=[]
    data=[]
    for i in range(len(data_file)):
        if tested_eye == data_file[i]['TestedEye'] and float(data_file[i]['Stimulus Frequency']) >= stim_freq[0] and float(data_file[i]['Stimulus Frequency']) < stim_freq[1]:
                waveform_time = data_file[i]['Reported Waveform'][0]
                waveform_uv = data_file[i]['Reported Waveform'][1]
                time, data = sep(waveform_time, waveform_uv)
    return time, data


def sensortype(data_file):
    s_type=' '
    if data_file[0]['ElectrodePackageType'] == 'ElectrodeType_LkcRimElectrodePair_SensorStrip':
        s_type= 'Normal'
    if data_file[0]['ElectrodePackageType'] == 'ElectrodeType_SmallSensorStrip':
        s_type = 'Small'
    return s_type



def labelling(ordered_data_file):
    # Check for disease or corrupt:
    i=0
    k=[1,2,3,4,5,6,7,8]
    for i in range(len(ordered_data_file)):
        normal='Normal'  + str(k[i])
        disease = 'Disease' + str(k[i])
        corrupt = 'Corrupt' + str(k[i])
        if values[disease] == True:
            ordered_data_file[i]['Label']='Disease'
        if values[corrupt] == True:
            ordered_data_file[i]['Label']='Corrupt'
        if values[normal] == True:
            ordered_data_file[i]['Label'] = 'Normal'
    return ordered_data_file


ERG_protocols= ['Dark 0.01','Dark 3.0', 'Light 3.0', 'Light 30']
ERG_protocols=ERG_protocols*2


def get_data(ordered_data_file,k):
    output_string= str(k+1) + '. ' + 'Tested eye: ' +str(ordered_data_file[k].get('TestedEye')) \
                        + ', ERG protocol: ' + ERG_protocols[k] \
                        + ', Stimulus Frequency: ' + str(ordered_data_file[k].get('Stimulus Frequency')) \
                        +  ', Label: '+ str(ordered_data_file[k].get('Label'))
    return output_string


def data_reorder(data_file):
    left_eye = []
    right_eye = []
    for i in range(len(data_file)):
        if data_file[i]['TestedEye']=='LeftEye':
            left_eye.append(data_file[i])
            #left_eye_stim.append(float(data_file[i]['Stimulus Frequency']))

        if data_file[i]['TestedEye'] =='RightEye':
            right_eye.append(data_file[i])
            #right_eye_stim.append(float(data_file[i]['Stimulus Frequency']))

    newlist_left = sorted(left_eye, key=itemgetter('Stimulus Frequency'))
    newlist_right = sorted(right_eye, key=itemgetter('Stimulus Frequency'))

    new_data= newlist_left + newlist_right

    return new_data


def subplotting(states):
    length = len(np.where(states)[0]) #3
    sb=[]
    for i in range(length):
        sb.append(length*100 + 10 + i+1) # [311 312 313]

    state_map = map(int, states)
    state_list = list(state_map) #[0 1 1 1]

    for i in range(len(state_list)):
        if state_list[i]==1:
            state_list[i]=sb[0]
            sb.pop(0)

    return state_list


# ---------------------- MATPLOTLIB FIGURE IN A TKINTER CANVAS + MATPLOTLIB TOOLBAR -------------------------#

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    #figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)
    figure_canvas_agg.get_tk_widget().pack(side='right', expand=1)

class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


# ---------------------- DEFINE LAYOUT -------------------------#
options = [[sg.Frame('Choose the eye',
                     [[sg.Checkbox('Left eye', key='Lefteye'),
                       sg.Checkbox('Right eye', key='Righteye')]], border_width=10)],
           [sg.Frame('Choose ERG recording',
                     [[sg.Checkbox('Dark-adapted 0.01 ERG (~0.5 Hz) ', key='D001'),
                       sg.Checkbox('Dark-adapted 3 ERG (~1 Hz)', key='D3')],
                      [sg.Checkbox('Light-adapted 3 ERG (~2 Hz)', key='L3'),
                       sg.Checkbox('Light-adapted 30 Hz flicker ERG (~28 Hz)', key='L30')]],
                       title_location='ne', background_color='lightyellow')],
           [sg.Button('Submit', font=('Times New Roman', 12))]]


label_data = [[sg.Frame('Label left eye recording',
                        [[sg.T('Dark-adapted 0.01 ERG:'),
                              sg.Radio('Normal', 'left1', key='Normal1', default=True),
                              sg.Radio('Disease', 'left1', key='Disease1', default=False),
                              sg.Radio('Corrupt/ improperly recorded', 'left1', key='Corrupt1', default=False)],
                        [sg.T('Dark-adapted 3 ERG: '),
                             sg.Radio('Normal', 'left2', key='Normal2', default=True),
                             sg.Radio('Disease', 'left2', key='Disease2'),
                             sg.Radio('Corrupt/ improperly recorded', 'left2',key='Corrupt2')],
                        [sg.T('Light-adapted 3 ERG: '),
                            sg.Radio('Normal', 'left3', key='Normal3', default=True),
                            sg.Radio('Disease', 'left3', key='Disease3'),
                            sg.Radio('Corrupt/ improperly recorded', 'left3',key='Corrupt3')],
                        [sg.T('Light-adapted 30 ERG: '),
                            sg.Radio('Normal', 'left4', key='Normal4', default=True),
                            sg.Radio('Disease', 'left4', key='Disease4'),
                            sg.Radio('Corrupt/ improperly recorded', 'left4',key='Corrupt4')]],
                     border_width=10)],

              [sg.Frame('Label right eye recording',
                        [[sg.T('Dark-adapted 0.01 ERG:'), sg.Radio('Normal', 'right1', key='Normal5', default=True),
                            sg.Radio('Disease', 'right1', key='Disease5'),
                            sg.Radio('Corrupt/ improperly recorded', 'right1', key='Corrupt5')],
                        [sg.T('Dark-adapted 3 ERG: '),
                            sg.Radio('Normal', 'right2', key='Normal6', default=True),
                            sg.Radio('Disease', 'right2', key='Disease6'),
                            sg.Radio('Corrupt/ improperly recorded', 'right2',key='Corrupt6')],
                        [sg.T('Light-adapted 3 ERG: '),
                            sg.Radio('Normal', 'right3', key='Normal7', default=True),
                            sg.Radio('Disease', 'right3', key='Disease7'),
                            sg.Radio('Corrupt/ improperly recorded', 'right3',key='Corrupt7')],
                        [sg.T('Light-adapted 30 ERG: '),
                            sg.Radio('Normal', 'right4', key='Normal8', default=True),
                            sg.Radio('Disease', 'right4', key='Disease8'),
                            sg.Radio('Corrupt/ improperly recorded', 'right4',key='Corrupt8')] ],
                     border_width=10)],
                        [sg.Button('Save', font=('Times New Roman', 12), key='Save')]]


#Table
headers = {'Data':[], 'Patient     ':[]}
patient_data =[['Patient ID','-'], ['Patients birthday','-'], ['Electrode type','-']]
headings = list(headers)


choices = [[sg.Text('Data:')],
           [sg.Table(values = patient_data, headings = headings, max_col_width=35,auto_size_columns=True,
                    display_row_numbers=False,justification='left',
                    alternating_row_color='lightyellow',
                    row_height=35,num_rows=3, key='-TABLE-', enable_events=True)],
            [sg.Frame('Select recording', layout=options)],
            [sg.Frame('Select recording', layout=label_data)],
           ]

items_chosen = [
        [sg.Canvas(key='-CANVAS-', size=(600,600))],
        [sg.Push(),sg.Button('Ok'), sg.Button('Exit')],
        [sg.Text("", size=(50, 3), key='options')],
                ]

plot_test=[
        [sg.B('Plot'), sg.B('Clear plot', key='Clear'), sg.B('Exit') ],
        [sg.T('Controls:')],
        [sg.Canvas(key='controls_cv')],
        [sg.T('Figure:')],

        [sg.Column(
        layout=[
            [sg.Canvas(key='fig_cv',
                       # it's important that you set this size
                       size=(400 *2, 400*1.1)
                       )]
        ],
        background_color='#DAE0E6',
        pad=(0, 0) )
           ],

        [sg.MLine(default_text=f'Automated classification:\n', size=(110, 6), auto_refresh=True, key='query')],

        ]



layout = [[sg.Menu(menu_def, tearoff=True)],[sg.Column(choices, element_justification='u'), sg.Column(plot_test,  element_justification='u')]]
window = sg.Window('Enlighter', layout, location=(0, 0),size=(1920,1080), font='Helvetica 9', resizable=True)




while True:
    event, values= window.read()
    if event is None or event == 'Exit':
        break

    if event == 'About...':
        sg.popup('Enlighter', 'Version 1.0', 'The enlighter development team was formed earlier this year to help '
                                             'detect patients with retinal disorders and thus support the early treatment of those suffering from this common disease')

    if event == 'Support':
        sg.popup('If you need any help please contact our Support Team', ' ', 'Data import and early  data processing:' ,' Anett Matuscsák <anettmatuscsak@gmail.com> ', ' ',
                 'GUI development:', ' Klaudia Csikós <csikos.klaudia2@gmail.com> ', ' ','Automated classification & Everything else:' , ' Ábel Petik <abelpetik@gmail.com > ')

    if event == 'Open':

        event = 'Clear'
        window['query'].update('Automated classification:\n')
        patient_text_output.clear()


        data_file, path_to_mydata =data_import.importer(return_path=True)
        i = 0
        k=0
        while i < len(data_file):
            if float(data_file[k]['Stimulus Frequency']) == 1.0 or str(data_file[k]['Stimulus Frequency']) == 'nan':
                data_file.pop(k)
            i = i + 1
            k=k+1

        patient_id=data_file[0]['PatientID']
        patient_birthday = data_file[0]['PatientBirthdate']
        electrode_type = sensortype(data_file)


        patient_data = [['Patient ID', patient_id], ['Patients birthday', patient_birthday], ['Electrode type', electrode_type]]
        window['-TABLE-'].update(patient_data)

        left_dark_001_time,left_dark_001_data= recordings_seperator(data_file, 'LeftEye' , [0.49, 0.5])
        left_dark_3_time, left_dark_3_data= recordings_seperator(data_file, 'LeftEye', [0.09, 1])
        left_light_3_time, left_light_3_data=recordings_seperator(data_file,'LeftEye', [1.9, 2.0])
        left_light_30_time, left_light_30_data = recordings_seperator(data_file, 'LeftEye', [27.0, 35.0])

        right_dark_001_time, right_dark_001_data = recordings_seperator(data_file, 'RightEye', [0.49, 0.5])
        right_dark_3_time, right_dark_3_data = recordings_seperator(data_file, 'RightEye',[0.09, 1])
        right_light_3_time, right_light_3_data = recordings_seperator(data_file, 'RightEye', [1.9, 2.0])
        right_light_30_time, right_light_30_data = recordings_seperator(data_file, 'RightEye', [27.0, 35.0])


        patient_text_output.append('Patient ID:' + str(patient_id))
        patient_text_output.append('Patient Birthdate:' + str(patient_birthday))
        patient_text_output.append('Electrode type:' + str(electrode_type))



        dicti= evaluate_single_patient(patient_id, os.path.split(path_to_mydata)[0])

        classification=''

        for protocol in dicti[electrode_type].keys():
            for eye in dicti[electrode_type][protocol].keys():
                if dicti[electrode_type][protocol][eye] > 4:
                    verdict = 'bad'
                elif dicti[electrode_type][protocol][eye] > 3:
                    verdict = 'suspicious'
                else:
                    verdict = 'good'
                classification += f'\n{protocol}, {eye}: {verdict}'

        print(classification)

        if len(classification)> 0:
            window['query'].update(f'Automated classification\n ------------------------------- {classification} \n-------------------------------')



    if event == 'Plot':
        # -------------------------------  MATPLOTLIB CODE --------------------------
        plt.figure(1)
        fig = plt.gcf()
        fig.clf()
        DPI = fig.get_dpi()
        fig.set_size_inches(404*2 / float(DPI), 404*1.1/ float(DPI))

        plt.title('ERG recording')
        plt.xlabel('ms')
        plt.ylabel('mV')
        plt.grid()


        # -------------------------------SUBPLOTTING-------------------------------

        states = [values['D001'], values['D3'], values['L3'], values['L30']]
        a = subplotting(states)[0] # e.g. [311 312 313]
        b = subplotting(states)[1]
        c = subplotting(states)[2]
        d = subplotting(states)[3]
        # -------------------------------Plotting the data-------------------------------

        if values['D001'] == True: #and values['Left eye'] == True:
            if a!=0:
                plt.subplot(a)
                plt.ylabel('mV')
            if values['Lefteye'] == True:
                plt.plot(left_dark_001_time, left_dark_001_data, color='green', label='left eye')
            if values['Righteye'] == True:
                plt.plot(right_dark_001_time, right_dark_001_data, color='grey', label='right eye')
            plt.legend()
            plt.title('Dark 0.01')

        if values['D3'] == True:
            if b!=0:
                plt.subplot(b)
                plt.ylabel('mV')
            if values['Lefteye'] == True:
                plt.plot(left_dark_3_time, left_dark_3_data, color='green', label='left eye')
            if values['Righteye'] == True:
                plt.plot(right_dark_3_time, right_dark_3_data, color='grey', label='right eye')
            plt.legend()
            plt.title('Dark 3')

        if values['L3'] == True:
            if c!=0:
                plt.subplot(c)
                plt.ylabel('mV')
            if values['Lefteye'] == True:
                plt.plot(left_light_3_time, left_light_3_data, color='green', label='left eye')
            if values['Righteye'] == True:
                plt.plot(right_light_3_time, right_light_3_data, color='grey', label='right eye')
            plt.legend()
            plt.title('Light 3')

        if values['L30'] == True:
            if d != 0:
                plt.subplot(d)
                #plt.xlabel('ms')
                plt.ylabel('mV')
            if values['Lefteye'] == True:
                plt.plot(left_light_30_time, left_light_30_data, color='green', label='left eye')
            if values['Righteye'] == True:
                plt.plot(right_light_30_time, right_light_30_data, color='grey', label='right eye')
            plt.legend()

            plt.title('Light 30')

        plt.xlabel('ms')

        #if values['D001'] == False and values['D3'] == False and values['L3'] == False and values['L30'] == False:
        #   plt.clf()

    # ------------------------------- Instead of plt.show()
        draw_figure_w_toolbar(window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

    # ------------------------------- Clearing the plot
    if event == 'Clear':  # If submit button is clicked display chosen values
        values['Righteye'] = False
        values['Lefteye'] = False

        values['D001'] = False
        values['D3'] = False
        values['L3'] = False
        values['L30'] = False

        a = 0
        b = 0
        c = 0
        d = 0

        plt.figure(1)
        plt.clf()
        fig = plt.gcf()
        DPI = fig.get_dpi()
        fig.set_size_inches(404*2/ float(DPI), 404*1.1/ float(DPI))

        plt.title('ERG recording')
        plt.xlabel('ms')
        plt.ylabel('mV')
        plt.grid()

        draw_figure_w_toolbar(window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)


    if event =='Save':
        labelled_data=""
        ordered_data_file = data_reorder(data_file)
        ordered_data_file = labelling(ordered_data_file)

        query = values['query'].rstrip()

        outfile= os.path.splitext(path_to_mydata)[0]+'.txt'
        print(outfile)
        with open(outfile, 'w') as outputfile:
            outputfile.write(" ".join(patient_text_output)+'\n')
        with open(outfile, 'a') as outputfile:
            outputfile.write(" " + '\n')
            outputfile.write("Manually labelled data " + '\n')
            outputfile.write("-------------------------------" + '\n')
            for k in range(len(ordered_data_file)):
                labelled_data = get_data(ordered_data_file, k)
                outputfile.write("".join(labelled_data) + '\n')
            outputfile.write("-------------------------------" + '\n')

            outputfile.write(" " + '\n')
            outputfile.write(query + '\n')

        data_import.save(os.path.splitext(path_to_mydata)[0]+'.pkl', ordered_data_file )



window.close()
