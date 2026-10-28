#!/usr/bin/env python
'''
Check that ghg files are valid zip files

History
-------
   * Written Oct 2026, Matthias Cuntz

'''
import glob
import os
import zipfile

domv = True  # True/False: Do/Do not move erroneous files to old/

# macOS NAS
ibase = '/Volumes/nancy-eef-prnas/Sites_ateliers/BD_Hesse'
# Windows NAS
ibase = 'Z:/Sites_ateliers/BD_Hesse'
# Windows local
ibase = 'J:/BD_Hesse/'

paths = [f'{ibase}/Donnees_brutes/2019/Auto/SmartfluxNT_2019',
         f'{ibase}/Donnees_brutes/2020/Auto/SmartfluxNT_2020',
         f'{ibase}/Donnees_brutes/2021/Auto/SmartfluxNT_2021',
         f'{ibase}/Donnees_brutes/2022/Auto/SmartfluxNT_2022',
         f'{ibase}/Donnees_brutes/2023/Auto/SmartfluxNT_2023/raw/',
         f'{ibase}/Donnees_brutes/2024/Auto/SmartfluxNT/raw/',
         f'{ibase}/Donnees_brutes/2025/Auto/SmartfluxNT/raw/',
         f'{ibase}/Donnees_brutes/2026/Auto/SmartfluxNT/raw/',]

paths = [f'{ibase}/Donnees_brutes/2023/Auto/SmartfluxNT_2023/raw/',
         f'{ibase}/Donnees_brutes/2024/Auto/SmartfluxNT/raw/',
         f'{ibase}/Donnees_brutes/2025/Auto/SmartfluxNT/raw/',
         f'{ibase}/Donnees_brutes/2026/Auto/SmartfluxNT/raw/',]

# paths = [f'{ibase}/Donnees_brutes/2022/Auto/SmartfluxNT_2022']

for ipath in paths:
    print(f'\n{ipath}')
    files = glob.glob(f'{ipath}/*.ghg')
    files.sort()

    badfiles = []
    for ff in files:
        bfile = os.path.basename(ff)
        print(bfile)
        try:
            with zipfile.ZipFile(ff) as zfile:
                dfile = ''
                for nn in zfile.namelist():
                    if nn.endswith('.data') and ('/' not in nn):
                        if dfile == '':
                            dfile = nn
                        else:
                            raise KeyError('check failed 1')
                if dfile == '':
                    raise KeyError('check failed 2')
                with zfile.open(dfile) as ff:
                    pass
        except (zipfile.BadZipFile, KeyError):
            bfile = os.path.basename(ff)
            print(f'    error {bfile}')
            badfiles.append(bfile)

    print(f'{len(badfiles)} erroneous files')
    print(badfiles)

    if domv:
        if len(badfiles) > 0:
            os.makedirs(f'{ipath}/old', exist_ok=True)
            for ff in badfiles:
                print(f'Move to old {ff}')
                os.rename(f'{ipath}/{ff}', f'{ipath}/old/{ff}')
