#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PATH="/content/drive/My Drive/Kandidat speciale/500 - Notebooks/models/all races_CV50/"


#### AIR ### 

AIR=True

file_name="COMPAS_dataset_OHE_std.csv"

full_file_path="/restricted/s164512/G2020-57-Aalborg-bias/COMPAS/Data/"+file_name



#titel_mitigation="original"
#titel_mitigation="DroppingD"
#titel_mitigation="Gender Swap"
#titel_mitigation="DI remove"
#titel_mitigation="LFR"
titel_mitigation="COMPAS"




PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN_ohe/models/"+titel_mitigation+"/"

dropping_D=True
gender_swap=False
DI_remove=False
LFR_mitigation=False #Sæt droppingD=True, men ikke fjern den fra X

y_col_name="is_recid"

X_col_names=['sex',
'age',
'juv_fel_count',
'juv_misd_count',
'juv_other_count',
'priors_count',
'race_African-American',
'race_Asian',
'race_Caucasian',
'race_Hispanic',
'race_Native American',
'race_Other',
'c_charge_desc_Abuse Without Great Harm',
'c_charge_desc_Accessory After the Fact',
'c_charge_desc_Agg Abuse Elderlly/Disabled Adult',
'c_charge_desc_Agg Assault Law Enforc Officer',
'c_charge_desc_Agg Assault W/int Com Fel Dome',
'c_charge_desc_Agg Battery Grt/Bod/Harm',
'c_charge_desc_Agg Fleeing and Eluding',
'c_charge_desc_Agg Fleeing/Eluding High Speed',
'c_charge_desc_Aggr Child Abuse-Torture,Punish',
'c_charge_desc_Aggrav Battery w/Deadly Weapon',
'c_charge_desc_Aggrav Child Abuse-Agg Battery',
'c_charge_desc_Aggrav Child Abuse-Causes Harm',
'c_charge_desc_Aggrav Stalking After Injunctn',
'c_charge_desc_Aggravated Assault',
'c_charge_desc_Aggravated Assault W/Dead Weap',
'c_charge_desc_Aggravated Assault W/dead Weap',
'c_charge_desc_Aggravated Assault W/o Firearm',
'c_charge_desc_Aggravated Assault w/Firearm',
'c_charge_desc_Aggravated Battery',
'c_charge_desc_Aggravated Battery (Firearm)',
'c_charge_desc_Aggravated Battery (Firearm/Actual Possession)',
'c_charge_desc_Aggravated Battery / Pregnant',
'c_charge_desc_Aggravated Battery On 65/Older',
'c_charge_desc_Aggress/Panhandle/Beg/Solict',
'c_charge_desc_Aide/Abet Prostitution Lewdness',
'c_charge_desc_Aiding Escape',
'c_charge_desc_Alcoholic Beverage Violation-FL',
'c_charge_desc_Armed Trafficking in Cannabis',
'c_charge_desc_Arson II (Vehicle)',
'c_charge_desc_Arson in the First Degree',
'c_charge_desc_Assault',
'c_charge_desc_Assault Law Enforcement Officer',
'c_charge_desc_Att Burgl Conv Occp',
'c_charge_desc_Att Burgl Struc/Conv Dwel/Occp',
'c_charge_desc_Att Burgl Unoccupied Dwel',
'c_charge_desc_Att Tamper w/Physical Evidence',
'c_charge_desc_Attempt Armed Burglary Dwell',
'c_charge_desc_Attempt Burglary (Struct)',
'c_charge_desc_Attempted Burg/Convey/Unocc',
'c_charge_desc_Attempted Burg/struct/unocc',
'c_charge_desc_Attempted Deliv Control Subst',
'c_charge_desc_Attempted Robbery  No Weapon',
'c_charge_desc_Attempted Robbery  Weapon',
'c_charge_desc_Attempted Robbery Firearm',
'c_charge_desc_Battery',
'c_charge_desc_Battery Emergency Care Provide',
'c_charge_desc_Battery On A Person Over 65',
'c_charge_desc_Battery On Fire Fighter',
'c_charge_desc_Battery On Parking Enfor Speci',
'c_charge_desc_Battery Spouse Or Girlfriend',
'c_charge_desc_Battery on Law Enforc Officer',
'c_charge_desc_Battery on a Person Over 65',
'c_charge_desc_Bribery Athletic Contests',
'c_charge_desc_Burgl Dwel/Struct/Convey Armed',
'c_charge_desc_Burglary Assault/Battery Armed',
'c_charge_desc_Burglary Conveyance Armed',
'c_charge_desc_Burglary Conveyance Assault/Bat',
'c_charge_desc_Burglary Conveyance Occupied',
'c_charge_desc_Burglary Conveyance Unoccup',
'c_charge_desc_Burglary Dwelling Armed',
'c_charge_desc_Burglary Dwelling Assault/Batt',
'c_charge_desc_Burglary Dwelling Occupied',
'c_charge_desc_Burglary Structure Assault/Batt',
'c_charge_desc_Burglary Structure Occupied',
'c_charge_desc_Burglary Structure Unoccup',
'c_charge_desc_Burglary Unoccupied Dwelling',
'c_charge_desc_Burglary With Assault/battery',
'c_charge_desc_Carjacking w/o Deadly Weapon',
'c_charge_desc_Carjacking with a Firearm',
'c_charge_desc_Carry Open/Uncov Bev In Pub',
'c_charge_desc_Carrying A Concealed Weapon',
'c_charge_desc_Carrying Concealed Firearm',
'c_charge_desc_Cash Item w/Intent to Defraud',
'c_charge_desc_Cause Anoth Phone Ring Repeat',
'c_charge_desc_Child Abuse',
'c_charge_desc_Compulsory Attendance Violation',
'c_charge_desc_Compulsory Sch Attnd Violation',
'c_charge_desc_Computer Pornography',
'c_charge_desc_Consp Traff Oxycodone  4g><14g',
'c_charge_desc_Consp Traff Oxycodone 28g><30k',
'c_charge_desc_Conspiracy Dealing Stolen Prop',
'c_charge_desc_Conspiracy to Deliver Cocaine',
'c_charge_desc_Consume Alcoholic Bev Pub',
'c_charge_desc_Contradict Statement',
'c_charge_desc_Contribute Delinquency Of A Minor',
'c_charge_desc_Corrupt Public Servant',
'c_charge_desc_Counterfeit Lic Plates/Sticker',
'c_charge_desc_Crim Attempt/Solic/Consp',
'c_charge_desc_Crim Attempt/Solicit/Consp',
'c_charge_desc_Crim Use Of Personal Id Info',
'c_charge_desc_Crim Use of Personal ID Info',
'c_charge_desc_Crimin Mischief Damage $1000+',
'c_charge_desc_Criminal Attempt 3rd Deg Felon',
'c_charge_desc_Criminal Mischief',
'c_charge_desc_Criminal Mischief Damage <$200',
'c_charge_desc_Criminal Mischief>$200<$1000',
'c_charge_desc_Crlty Twrd Child Urge Oth Act',
'c_charge_desc_Cruelty Toward Child',
'c_charge_desc_Cruelty to Animals',
'c_charge_desc_Culpable Negligence',
'c_charge_desc_D.U.I. Serious Bodily Injury',
'c_charge_desc_DOC/Cause Public Danger',
'c_charge_desc_DUI - Enhanced',
'c_charge_desc_DUI - Property Damage/Personal Injury',
'c_charge_desc_DUI Blood Alcohol Above 0.20',
'c_charge_desc_DUI Level 0.15 Or Minor In Veh',
'c_charge_desc_DUI Property Damage/Injury',
'c_charge_desc_DUI- Enhanced',
'c_charge_desc_DUI/Property Damage/Persnl Inj',
'c_charge_desc_DWI w/Inj Susp Lic / Habit Off',
'c_charge_desc_DWLS Canceled Disqul 1st Off',
'c_charge_desc_DWLS Susp/Cancel Revoked',
'c_charge_desc_Dealing In Stolen Property',
'c_charge_desc_Dealing in Stolen Property',
'c_charge_desc_Defrauding Innkeeper',
'c_charge_desc_Defrauding Innkeeper $300/More',
'c_charge_desc_Del 3,4 Methylenedioxymethcath',
'c_charge_desc_Del Cannabis At/Near Park',
'c_charge_desc_Del Cannabis For Consideration',
'c_charge_desc_Del Morphine at/near Park',
'c_charge_desc_Del of JWH-250 2-Methox 1-Pentyl',
'c_charge_desc_Deliver 3,4 Methylenediox',
'c_charge_desc_Deliver Alprazolam',
'c_charge_desc_Deliver Cannabis',
'c_charge_desc_Deliver Cannabis 1000FTSch',
'c_charge_desc_Deliver Cocaine',
'c_charge_desc_Deliver Cocaine 1000FT Church',
'c_charge_desc_Deliver Cocaine 1000FT Park',
'c_charge_desc_Deliver Cocaine 1000FT School',
'c_charge_desc_Deliver Cocaine 1000FT Store',
'c_charge_desc_Delivery Of Drug Paraphernalia',
'c_charge_desc_Delivery of 5-Fluoro PB-22',
'c_charge_desc_Delivery of Heroin',
'c_charge_desc_Depriv LEO of Protect/Communic',
'c_charge_desc_Discharge Firearm From Vehicle',
'c_charge_desc_Disorderly Conduct',
'c_charge_desc_Disorderly Intoxication',
'c_charge_desc_Disrupting School Function',
'c_charge_desc_Drivg While Lic Suspd/Revk/Can',
'c_charge_desc_Driving License Suspended',
'c_charge_desc_Driving Under The Influence',
'c_charge_desc_Driving While License Revoked',
'c_charge_desc_Escape',
'c_charge_desc_Exhibition Weapon School Prop',
'c_charge_desc_Expired DL More Than 6 Months',
'c_charge_desc_Exploit Elderly Person 20-100K',
'c_charge_desc_Exposes Culpable Negligence',
'c_charge_desc_Extradition/Defendants',
'c_charge_desc_Fabricating Physical Evidence',
'c_charge_desc_Fail Obey Driv Lic Restrictions',
'c_charge_desc_Fail Register Career Offender',
'c_charge_desc_Fail Register Vehicle',
'c_charge_desc_Fail Sex Offend Report Bylaw',
'c_charge_desc_Fail To Obey Police Officer',
'c_charge_desc_Fail To Redeliv Hire/Leas Prop',
'c_charge_desc_Fail To Redeliver Hire Prop',
'c_charge_desc_Fail To Secure Load',
'c_charge_desc_Failure To Pay Taxi Cab Charge',
'c_charge_desc_Failure To Return Hired Vehicle',
'c_charge_desc_False 911 Call',
'c_charge_desc_False Bomb Report',
'c_charge_desc_False Imprisonment',
'c_charge_desc_False Info LEO During Invest',
'c_charge_desc_False Motor Veh Insurance Card',
'c_charge_desc_False Name By Person Arrest',
'c_charge_desc_False Ownership Info/Pawn Item',
'c_charge_desc_Falsely Impersonating Officer',
'c_charge_desc_Fel Drive License Perm Revoke',
'c_charge_desc_Felon in Pos of Firearm or Amm',
'c_charge_desc_Felony Batt(Great Bodily Harm)',
'c_charge_desc_Felony Battery',
'c_charge_desc_Felony Battery (Dom Strang)',
'c_charge_desc_Felony Battery w/Prior Convict',
'c_charge_desc_Felony Committing Prostitution',
'c_charge_desc_Felony DUI (level 3)',
'c_charge_desc_Felony DUI - Enhanced',
'c_charge_desc_Felony Driving While Lic Suspd',
'c_charge_desc_Felony Petit Theft',
'c_charge_desc_Felony/Driving Under Influence',
'c_charge_desc_Fighting/Baiting Animals',
'c_charge_desc_Flee/Elude LEO-Agg Flee Unsafe',
'c_charge_desc_Fleeing Or Attmp Eluding A Leo',
'c_charge_desc_Fleeing or Eluding a LEO',
'c_charge_desc_Forging Bank Bills/Promis Note',
'c_charge_desc_Fraud Obtain Food or Lodging',
'c_charge_desc_Fraudulent Use of Credit Card',
'c_charge_desc_Gambling/Gamb Paraphernalia',
'c_charge_desc_Giving False Crime Report',
'c_charge_desc_Grand Theft (Motor Vehicle)',
'c_charge_desc_Grand Theft (motor Vehicle)',
'c_charge_desc_Grand Theft Dwell Property',
'c_charge_desc_Grand Theft Firearm',
'c_charge_desc_Grand Theft In The 3Rd Degree',
'c_charge_desc_Grand Theft in the 1st Degree',
'c_charge_desc_Grand Theft in the 3rd Degree',
'c_charge_desc_Grand Theft of a Fire Extinquisher',
'c_charge_desc_Grand Theft of the 2nd Degree',
'c_charge_desc_Grand Theft on 65 Yr or Older',
'c_charge_desc_Harass Witness/Victm/Informnt',
'c_charge_desc_Harm Public Servant Or Family',
'c_charge_desc_Hiring with Intent to Defraud',
'c_charge_desc_Imperson Public Officer or Emplyee',
'c_charge_desc_Insurance Fraud',
'c_charge_desc_Interfere W/Traf Cont Dev RR',
'c_charge_desc_Interference with Custody',
'c_charge_desc_Intoxicated/Safety Of Another',
'c_charge_desc_Introduce Contraband Into Jail',
'c_charge_desc_Issuing a Worthless Draft',
'c_charge_desc_Kidnapping / Domestic Violence',
'c_charge_desc_Lease For Purpose Trafficking',
'c_charge_desc_Leave Acc/Attend Veh/More $50',
'c_charge_desc_Leave Accd/Attend Veh/Less $50',
'c_charge_desc_Leaving Acc/Unattended Veh',
'c_charge_desc_Leaving the Scene of Accident',
'c_charge_desc_Lewd Act Presence Child 16-',
'c_charge_desc_Lewd or Lascivious Molestation',
'c_charge_desc_Lewd/Lasc Battery Pers 12+/<16',
'c_charge_desc_Lewd/Lasc Exhib Presence <16yr',
'c_charge_desc_Lewd/Lasciv Molest Elder Persn',
'c_charge_desc_Lewdness Violation',
'c_charge_desc_License Suspended Revoked',
'c_charge_desc_Littering',
'c_charge_desc_Live on Earnings of Prostitute',
'c_charge_desc_Lve/Scen/Acc/Veh/Prop/Damage',
'c_charge_desc_Manage Busn W/O City Occup Lic',
'c_charge_desc_Manslaughter W/Weapon/Firearm',
'c_charge_desc_Manufacture Cannabis',
'c_charge_desc_Misuse Of 911 Or E911 System',
'c_charge_desc_Money Launder 100K or More Dols',
'c_charge_desc_Murder In 2nd Degree W/firearm',
'c_charge_desc_Murder in 2nd Degree',
'c_charge_desc_Murder in the First Degree',
'c_charge_desc_Neglect Child / Bodily Harm',
'c_charge_desc_Neglect Child / No Bodily Harm',
'c_charge_desc_Neglect/Abuse Elderly Person',
'c_charge_desc_Obstruct Fire Equipment',
'c_charge_desc_Obstruct Officer W/Violence',
'c_charge_desc_Obtain Control Substance By Fraud',
'c_charge_desc_Offer Agree Secure For Lewd Act',
'c_charge_desc_Offer Agree Secure/Lewd Act',
'c_charge_desc_Offn Against Intellectual Prop',
'c_charge_desc_Open Carrying Of Weapon',
'c_charge_desc_Oper Motorcycle W/O Valid DL',
'c_charge_desc_Operating W/O Valid License',
'c_charge_desc_Opert With Susp DL 2ND Offense',
'c_charge_desc_Opert With Susp DL 2nd Offens',
'c_charge_desc_PL/Unlaw Use Credit Card',
'c_charge_desc_Petit Theft',
'c_charge_desc_Petit Theft $100- $300',
'c_charge_desc_Pos Cannabis For Consideration',
'c_charge_desc_Pos Cannabis W/Intent Sel/Del',
'c_charge_desc_Pos Methylenedioxymethcath W/I/D/S',
'c_charge_desc_Poss 3,4 MDMA (Ecstasy)',
'c_charge_desc_Poss Alprazolam W/int Sell/Del',
'c_charge_desc_Poss Anti-Shoplifting Device',
'c_charge_desc_Poss Cntrft Contr Sub w/Intent',
'c_charge_desc_Poss Cocaine/Intent To Del/Sel',
'c_charge_desc_Poss Contr Subst W/o Prescript',
'c_charge_desc_Poss Counterfeit Payment Inst',
'c_charge_desc_Poss Drugs W/O A Prescription',
'c_charge_desc_Poss F/Arm Delinq',
'c_charge_desc_Poss Firearm W/Altered ID#',
'c_charge_desc_Poss Meth/Diox/Meth/Amp (MDMA)',
'c_charge_desc_Poss Of 1,4-Butanediol',
'c_charge_desc_Poss Of Controlled Substance',
'c_charge_desc_Poss Of RX Without RX',
'c_charge_desc_Poss Oxycodone W/Int/Sell/Del',
'c_charge_desc_Poss Pyrrolidinobutiophenone',
'c_charge_desc_Poss Pyrrolidinovalerophenone',
'c_charge_desc_Poss Pyrrolidinovalerophenone W/I/D/S',
'c_charge_desc_Poss Similitude of Drivers Lic',
'c_charge_desc_Poss Tetrahydrocannabinols',
'c_charge_desc_Poss Trifluoromethylphenylpipe',
'c_charge_desc_Poss Unlaw Issue Driver Licenc',
'c_charge_desc_Poss Unlaw Issue Id',
'c_charge_desc_Poss Wep Conv Felon',
'c_charge_desc_Poss of Cocaine W/I/D/S 1000FT Park',
'c_charge_desc_Poss of Firearm by Convic Felo',
'c_charge_desc_Poss of Methylethcathinone',
'c_charge_desc_Poss of Vessel w/Altered ID NO',
'c_charge_desc_Poss/Sell/Del Cocaine 1000FT Sch',
'c_charge_desc_Poss/Sell/Del/Man Amobarbital',
'c_charge_desc_Poss/Sell/Deliver Clonazepam',
'c_charge_desc_Poss/pur/sell/deliver Cocaine',
'c_charge_desc_Poss3,4 Methylenedioxymethcath',
'c_charge_desc_Posses/Disply Susp/Revk/Frd DL',
'c_charge_desc_Possess Cannabis 1000FTSch',
'c_charge_desc_Possess Cannabis/20 Grams Or Less',
'c_charge_desc_Possess Controlled Substance',
'c_charge_desc_Possess Countrfeit Credit Card',
'c_charge_desc_Possess Drug Paraphernalia',
'c_charge_desc_Possess Mot Veh W/Alt Vin #',
'c_charge_desc_Possess Tobacco Product Under 18',
'c_charge_desc_Possess Weapon On School Prop',
'c_charge_desc_Possess w/I/Utter Forged Bills',
'c_charge_desc_Possess/Use Weapon 1 Deg Felon',
'c_charge_desc_Possession Burglary Tools',
'c_charge_desc_Possession Child Pornography',
'c_charge_desc_Possession Firearm School Prop',
'c_charge_desc_Possession Of 3,4Methylenediox',
'c_charge_desc_Possession Of Alprazolam',
'c_charge_desc_Possession Of Amphetamine',
'c_charge_desc_Possession Of Anabolic Steroid',
'c_charge_desc_Possession Of Buprenorphine',
'c_charge_desc_Possession Of Carisoprodol',
'c_charge_desc_Possession Of Clonazepam',
'c_charge_desc_Possession Of Cocaine',
'c_charge_desc_Possession Of Diazepam',
'c_charge_desc_Possession Of Fentanyl',
'c_charge_desc_Possession Of Heroin',
'c_charge_desc_Possession Of Lorazepam',
'c_charge_desc_Possession Of Methamphetamine',
'c_charge_desc_Possession Of Paraphernalia',
'c_charge_desc_Possession Of Phentermine',
'c_charge_desc_Possession of Alcohol Under 21',
'c_charge_desc_Possession of Benzylpiperazine',
'c_charge_desc_Possession of Butylone',
'c_charge_desc_Possession of Cannabis',
'c_charge_desc_Possession of Cocaine',
'c_charge_desc_Possession of Codeine',
'c_charge_desc_Possession of Ethylone',
'c_charge_desc_Possession of Hydrocodone',
'c_charge_desc_Possession of Hydromorphone',
'c_charge_desc_Possession of LSD',
'c_charge_desc_Possession of Methadone',
'c_charge_desc_Possession of Morphine',
'c_charge_desc_Possession of Oxycodone',
'c_charge_desc_Possession of XLR11',
'c_charge_desc_Present Proof of Invalid Insur',
'c_charge_desc_Principal In The First Degree',
'c_charge_desc_Prostitution',
'c_charge_desc_Prostitution/Lewd Act Assignation',
'c_charge_desc_Prostitution/Lewdness/Assign',
'c_charge_desc_Prowling/Loitering',
'c_charge_desc_Purchase Cannabis',
'c_charge_desc_Purchase Of Cocaine',
'c_charge_desc_Purchase/P/W/Int Cannabis',
'c_charge_desc_Purchasing Of Alprazolam',
'c_charge_desc_Reckless Driving',
'c_charge_desc_Refuse Submit Blood/Breath Test',
'c_charge_desc_Refuse to Supply DNA Sample',
'c_charge_desc_Resist Officer w/Violence',
'c_charge_desc_Resist/Obstruct W/O Violence',
'c_charge_desc_Restraining Order Dating Viol',
'c_charge_desc_Retail Theft $300 1st Offense',
'c_charge_desc_Retail Theft $300 2nd Offense',
'c_charge_desc_Ride Tri-Rail Without Paying',
'c_charge_desc_Robbery / No Weapon',
'c_charge_desc_Robbery / Weapon',
'c_charge_desc_Robbery Sudd Snatch No Weapon',
'c_charge_desc_Robbery W/Deadly Weapon',
'c_charge_desc_Robbery W/Firearm',
'c_charge_desc_Sale/Del Cannabis At/Near Scho',
'c_charge_desc_Sale/Del Counterfeit Cont Subs',
'c_charge_desc_Sel Etc/Pos/w/Int Contrft Schd',
'c_charge_desc_Sel/Pur/Mfr/Del Control Substa',
'c_charge_desc_Sell Cannabis',
'c_charge_desc_Sell Conterfeit Cont Substance',
'c_charge_desc_Sell or Offer for Sale Counterfeit Goods',
'c_charge_desc_Sell/Man/Del Pos/w/int Heroin',
'c_charge_desc_Sex Batt Faml/Cust Vict 12-17Y',
'c_charge_desc_Sex Battery Deft 18+/Vict 11-',
'c_charge_desc_Sex Offender Fail Comply W/Law',
'c_charge_desc_Sexual Battery / Vict 12 Yrs +',
'c_charge_desc_Sexual Performance by a Child',
'c_charge_desc_Shoot In Occupied Dwell',
'c_charge_desc_Shoot Into Vehicle',
'c_charge_desc_Simulation of Legal Process',
'c_charge_desc_Solic to Commit Battery',
'c_charge_desc_Solicit Deliver Cocaine',
'c_charge_desc_Solicit Purchase Cocaine',
'c_charge_desc_Solicit To Deliver Cocaine',
'c_charge_desc_Solicitation On Felony 3 Deg',
'c_charge_desc_Soliciting For Prostitution',
'c_charge_desc_Sound Articles Over 100',
'c_charge_desc_Stalking',
'c_charge_desc_Stalking (Aggravated)',
'c_charge_desc_Strong Armed  Robbery',
'c_charge_desc_Structuring Transactions',
'c_charge_desc_Susp Drivers Lic 1st Offense',
'c_charge_desc_Tamper With Victim',
'c_charge_desc_Tamper With Witness',
'c_charge_desc_Tamper With Witness/Victim/CI',
'c_charge_desc_Tampering With Physical Evidence',
'c_charge_desc_Tampering with a Victim',
'c_charge_desc_Theft',
'c_charge_desc_Theft/To Deprive',
'c_charge_desc_Threat Public Servant',
'c_charge_desc_Throw Deadly Missile Into Veh',
'c_charge_desc_Throw In Occupied Dwell',
'c_charge_desc_Throw Missile Into Pub/Priv Dw',
'c_charge_desc_Traff In Cocaine <400g>150 Kil',
'c_charge_desc_Traffic Counterfeit Cred Cards',
'c_charge_desc_Traffick Amphetamine 28g><200g',
'c_charge_desc_Traffick Hydrocodone   4g><14g',
'c_charge_desc_Traffick Oxycodone     4g><14g',
'c_charge_desc_Trans/Harm/Material to a Minor',
'c_charge_desc_Trespass On School Grounds',
'c_charge_desc_Trespass Other Struct/Conve',
'c_charge_desc_Trespass Private Property',
'c_charge_desc_Trespass Property w/Dang Weap',
'c_charge_desc_Trespass Struct/Convey Occupy',
'c_charge_desc_Trespass Struct/Conveyance',
'c_charge_desc_Trespass Structure w/Dang Weap',
'c_charge_desc_Trespass Structure/Conveyance',
'c_charge_desc_Trespassing/Construction Site',
'c_charge_desc_Tresspass Struct/Conveyance',
'c_charge_desc_Tresspass in Structure or Conveyance',
'c_charge_desc_Unauth C/P/S Sounds>1000/Audio',
'c_charge_desc_Unauth Poss ID Card or DL',
'c_charge_desc_Unauthorized Interf w/Railroad',
'c_charge_desc_Unemployment Compensatn Fraud',
'c_charge_desc_Unl/Disturb Education/Instui',
'c_charge_desc_Unlaw Lic Use/Disply Of Others',
'c_charge_desc_Unlaw LicTag/Sticker Attach',
'c_charge_desc_Unlaw Use False Name/Identity',
'c_charge_desc_Unlawful Conveyance of Fuel',
'c_charge_desc_Unlawful Use Of Police Badges',
'c_charge_desc_Unlicensed Telemarketing',
'c_charge_desc_Use Computer for Child Exploit',
'c_charge_desc_Use Of 2 Way Device To Fac Fel',
'c_charge_desc_Use Scanning Device to Defraud',
'c_charge_desc_Use of Anti-Shoplifting Device',
'c_charge_desc_Uttering Forged Bills',
'c_charge_desc_Uttering Forged Credit Card',
'c_charge_desc_Uttering Worthless Check +$150',
'c_charge_desc_Uttering a Forged Instrument',
'c_charge_desc_Video Voyeur-<24Y on Child >16',
'c_charge_desc_Viol Injunct Domestic Violence',
'c_charge_desc_Viol Injunction Protect Dom Vi',
'c_charge_desc_Viol Pretrial Release Dom Viol',
'c_charge_desc_Viol Prot Injunc Repeat Viol',
'c_charge_desc_Violation License Restrictions',
'c_charge_desc_Violation Of Boater Safety Id',
'c_charge_desc_Violation of Injunction Order/Stalking/Cyberstalking',
'c_charge_desc_Voyeurism',
'c_charge_desc_arrest case no charge',
'c_charge_degree_F',
'c_charge_degree_M']
#X_col_names = [col for col in X_col_names if col not in leave_out ]

procted_col_name="race"


###### COMPASS ####

#AIR=False

#titel_mitigation="testCOMPAS"
#PATH_orig="/restricted/s164512/G2020-57-Aalborg-bias/lau/FFNN/models/"+titel_mitigation+"/"

#full_file_path = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'

#y_col_name="is_recid"
#X_col_names=['remember_index','sex','age','race', 'juv_fel_count','juv_misd_count','juv_other_count','priors_count',"c_charge_desc","c_charge_degree"]

#procted_col_name="race"


# In[2]:


def LFR_custom(df_train,y_train,lfr=None):
    from aif360.algorithms.preprocessing import LFR
    from aif360.datasets import BinaryLabelDataset
    
    df_train=pd.concat([df_train,y_train],axis=1)
    
    X_col_names_f=['Gender', 'BirthYear', 'LoanPeriod', 'NumberAts']
    df2_all=df_train.drop(columns=X_col_names_f).copy() #Gemmer alle kolonner, undtagen numerical og gender
    df2=df_train[X_col_names_f+["Fall"]].copy() #Gem kun numerical features
    df2_gender=df_train["Gender"].copy() #Gemmer bare gender
    
    
    #Create the binarylabeldataset
    df_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=df2,
                                label_names=['Fall'],
                                protected_attribute_names=["Gender"],
                                unprivileged_protected_attributes=['0'])
    #Define the DI remover
    if lfr is None:
        lfr = LFR(privileged_groups=[{"Gender": 1}], 
                                    unprivileged_groups=[{"Gender": 0}])
        rp_df = lfr.fit_transform(df_BLD)
    else:
        rp_df = lfr.transform(df_BLD)
        

    #Save the columnnames
    all_col_names=df_BLD.feature_names+df_BLD.label_names
        
        
    
    #Save repaired data as pandas DF
    rp_df_pd = pd.DataFrame(np.hstack([rp_df.features,rp_df.labels]),columns=all_col_names) 
    
    #Somehow gender is also transformed! So we drop it! DETTE SKAL VI NOK LIGE HOLDE ØJE MED
    ###OBS!#####
    rp_df_pd = rp_df_pd.drop(columns=["Gender"])
    #rp_df_pd = pd.concat([rp_df_pd,df2_gender],axis=1)

    ##########
    
    
    #Concatenate the non-numerical columns
    transformed_data = pd.concat ([rp_df_pd,df2_all], axis=1)
    
    
    transformed_data=transformed_data.drop(columns=["Fall"])
    
    return transformed_data,lfr


# In[3]:


def DI_remove_custom(df_train,RP_level=1.0):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.datasets import BinaryLabelDataset
    X_col_names_f=['Gender', 'BirthYear', 'LoanPeriod', 'NumberAts']
    df2_all=df_train.drop(columns=X_col_names_f).copy() #Gemmer alle kolonner, undtagen numerical og gender
    df2=df_train[X_col_names_f].copy() #Gem kun numerical features
    
    df2["dummy"]=1 # this is a dummy variable, since DI remover dont use y. 
    
    #Create the binarylabeldataset
    df_BLD = BinaryLabelDataset(favorable_label='1',
                                unfavorable_label='0',
                                df=df2,
                                label_names=['dummy'],
                                protected_attribute_names=["Gender"],
                                unprivileged_protected_attributes=['0'])
    #Define the DI remover
    di = DisparateImpactRemover(repair_level=RP_level)
    #Save the columnnames
    all_col_names=df_BLD.feature_names+df_BLD.label_names
    #Reparing the data
    rp_df = di.fit_transform(df_BLD)  
    #Save repaired data as pandas DF
    rp_df_pd = pd.DataFrame(np.hstack([rp_df.features,rp_df.labels]),columns=all_col_names) 
    #Concatenate the non-numerical columns
    transformed_data = pd.concat ([rp_df_pd,df2_all], axis=1)
    
    
    transformed_data_train=transformed_data.drop(columns=["dummy"])

    
    return transformed_data_train


# In[4]:


n_nodes=2000


batch_size=40
epochs=200
p_drop=0.5

optim_type="Adam" #SGD
lr=0.001 #0.001 er godt
wd=0.05       




#0.01 er godt til AIR ny!!


# In[5]:


import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.datasets as datasets
from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from IPython.display import clear_output


import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#from google.colab import drive
from sklearn.model_selection import KFold

from datetime import datetime

import pytz
import random

import os
from sklearn.model_selection import StratifiedKFold


# In[6]:


plt.plot([0,0])


# In[7]:


from utils_Copy import *


# In[8]:


def loss_fn(target,predictions):
    criterion = nn.BCELoss()
    loss_out = criterion(predictions, target)
    return loss_out


# In[9]:



def accuracy(true,pred):
    acc = (true.float().round() == pred.float().round()).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))

def get_test():
    avg_loss_ts = 0
    avg_acc_ts=0
    model.eval()  # train mode
    for X_batch, Y_batch in data_ts:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)


        # forward
        Y_pred = model(X_batch.float()) 
        loss = loss_fn(Y_batch.float(), Y_pred.squeeze()) 

        # calculate metrics to show the user
        avg_loss_ts += loss / len(data_ts)
        avg_acc_ts+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_ts)
    #toc = time()

    return avg_loss_ts, avg_acc_ts

def get_all_time_low(all_time,new_val):
    if all_time>new_val:
        return new_val
    else:
        return all_time
    
def get_all_time_high(all_time,new_val):
    if all_time<new_val:
        return new_val
    else:
        return all_time


# In[10]:




class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fully_connected1 = nn.Sequential(
            nn.Linear(n_feat,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew1 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )

        self.fully_connectednew2 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )
        self.fully_connectednew3 = nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(n_nodes),
            nn.Dropout(p_drop)
            )


        self.fully_connected2 = nn.Sequential(
            nn.Linear(n_nodes,output_dim),
            #nn.Softmax(dim = 1)
            nn.Sigmoid()

            )

    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        #x = x.view(x.size(0),-1)
        x = self.fully_connected1(x)
        x = self.fully_connectednew(x)
        x = self.fully_connectednew1(x)
        x = self.fully_connectednew2(x)
        x = self.fully_connectednew3(x)
        x = self.fully_connected2(x)
        return x


# In[11]:


def custom_create_indexes(df,n,seed,strat=False,y_col=None):
    list_of_index=[]
    
    
    
    if strat==False:
        kf=KFold(n_splits=n, random_state=seed, shuffle=True)
        
        
        for train_index, test_index in kf.split(df):
            list_of_index.append(test_index)

        tr_val_ts_indexes=[
        #[[train],[validate],[test]]

        [[*list_of_index[0],*list_of_index[1],*list_of_index[2]],list_of_index[3],list_of_index[4]],
        [[*list_of_index[4],*list_of_index[0],*list_of_index[1]],list_of_index[2],list_of_index[3]],
        [[*list_of_index[3],*list_of_index[4],*list_of_index[0]],list_of_index[1],list_of_index[2]],
        [[*list_of_index[2],*list_of_index[3],*list_of_index[4]],list_of_index[0],list_of_index[1]],
        [[*list_of_index[1],*list_of_index[2],*list_of_index[3]],list_of_index[4],list_of_index[0]],

        ]
    
    
    
    
    else:
        kf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)
    
    
        for train_index, test_index in kf.split(df,df[y_col]):
            list_of_index.append(test_index)

        tr_val_ts_indexes=[
        #[[train],[validate],[test]]

        [[*list_of_index[0],*list_of_index[1],*list_of_index[2]],list_of_index[3],list_of_index[4]],
        [[*list_of_index[4],*list_of_index[0],*list_of_index[1]],list_of_index[2],list_of_index[3]],
        [[*list_of_index[3],*list_of_index[4],*list_of_index[0]],list_of_index[1],list_of_index[2]],
        [[*list_of_index[2],*list_of_index[3],*list_of_index[4]],list_of_index[0],list_of_index[1]],
        [[*list_of_index[1],*list_of_index[2],*list_of_index[3]],list_of_index[4],list_of_index[0]],

        ]
    
    return tr_val_ts_indexes


# In[12]:


modelcounter=0
for custom_seed in range(1,11):

    

    torch.manual_seed(custom_seed)
    random.seed(custom_seed)
    np.random.seed(custom_seed)
    

    df2 = pd.read_csv(full_file_path)



   
    X=df2[X_col_names]

    if dropping_D==True:
        y=df2[[y_col_name,procted_col_name]]
    else:
        y=df2[[y_col_name]]

    #https://stackoverflow.com/questions/11587782/creating-dummy-variables-in-pandas-for-python

    if AIR==False:
        just_dummies=pd.get_dummies(X[['sex',"race","c_charge_desc","c_charge_degree"]])
        X = pd.concat([X, just_dummies], axis=1) 
        X=X.drop(['sex',"race","c_charge_desc","c_charge_degree"] ,axis=1)


    
    tr_val_ts_indexes= custom_create_indexes(X,5,custom_seed)
    #tr_val_ts_indexes= custom_create_indexes(df2,5,custom_seed,True,y_col_name) #stratify

    #i=0
    for mini_loop in range(len(tr_val_ts_indexes)):
        print("Running overall number "+str(modelcounter))
         
        
        X_train_pd, y_train_pd = X.iloc[tr_val_ts_indexes[mini_loop][0]], y.iloc[tr_val_ts_indexes[mini_loop][0]]
        X_val_pd, y_val_pd = X.iloc[tr_val_ts_indexes[mini_loop][1]], y.iloc[tr_val_ts_indexes[mini_loop][1]]
        X_test_pd, y_test_pd = X.iloc[tr_val_ts_indexes[mini_loop][2]], y.iloc[tr_val_ts_indexes[mini_loop][2]]
        
        
        seedName="model"+str(modelcounter)

        PATH=PATH_orig+seedName+"/"
        print(PATH)

        #Make dir to files
        if not os.path.exists(PATH):
            os.makedirs(PATH)
            print("Created new path!: ",PATH)
        
        
        if gender_swap==True:
            X_train_pd_copy=X_train_pd.copy()
            y_train_pd_copy=y_train_pd.copy()
            
            X_train_pd_copy["Gender"]=(X_train_pd_copy["Gender"]-1)*(-1)
            
            X_train_pd=pd.concat([X_train_pd,X_train_pd_copy])
            
            y_train_pd=pd.concat([y_train_pd,y_train_pd_copy])
            
            
            
            
        if DI_remove==True:
            X_train_pd=DI_remove_custom(X_train_pd.reset_index(drop=True))
            X_val_pd=DI_remove_custom(X_val_pd.reset_index(drop=True))
            X_test_pd=DI_remove_custom(X_test_pd.reset_index(drop=True))
            
            y_train_pd=y_train_pd.reset_index(drop=True)
            
            y_val_pd=y_val_pd.reset_index(drop=True)
            
            y_test_pd=y_test_pd.reset_index(drop=True)
            
            
        if LFR_mitigation==True:
            
            
            X_train_pd,lfr=LFR_custom(X_train_pd.reset_index(drop=True),
                                  y_train_pd.reset_index(drop=True).drop(columns=["Gender"]),
                                 lfr=None
                                 )
            X_val_pd,lfr=LFR_custom(X_val_pd.reset_index(drop=True),
                                  y_val_pd.reset_index(drop=True).drop(columns=["Gender"]),
                                 lfr=lfr
                                 )
            X_test_pd,lfr=LFR_custom(X_test_pd.reset_index(drop=True),
                                  y_test_pd.reset_index(drop=True).drop(columns=["Gender"]),
                                 lfr=lfr
                                 )
            
            y_train_pd=y_train_pd.reset_index(drop=True)
            
            y_val_pd=y_val_pd.reset_index(drop=True)
            
            y_test_pd=y_test_pd.reset_index(drop=True)
            
            
            
        X_train, y_train = X_train_pd, y_train_pd
        X_val, y_val = X_val_pd, y_val_pd
        X_test, y_test = X_test_pd, y_test_pd


        #Save as numpy array for the DATALOADER (PyTorch)
        
        if LFR_mitigation==True:
            temp_col_name_LFR=[name for name in X_col_names if name not in ["Gender"]]
            X_train=np.array(X_train[temp_col_name_LFR])
            y_train=np.array(y_train[y_col_name])

            X_val=np.array(X_val[temp_col_name_LFR])
            y_val=np.array(y_val[y_col_name])

            X_test=np.array(X_test[temp_col_name_LFR])
            y_test=np.array(y_test[y_col_name])
        
        
        else:
            
            X_train=np.array(X_train[X_col_names])
            y_train=np.array(y_train[y_col_name])

            X_val=np.array(X_val[X_col_names])
            y_val=np.array(y_val[y_col_name])

            X_test=np.array(X_test[X_col_names])
            y_test=np.array(y_test[y_col_name])


       # print("X_train shape: {}".format(X_train.shape))
       # print("y_train shape: {}".format(y_train.shape))

        #print("X_val shape: {}".format(X_val.shape))
        #print("y_val shape: {}".format(y_val.shape))

        #print("X_test shape: {}".format(X_test.shape))
        #print("y_test shape: {}".format(y_test.shape))




        #n_feat=X_train.shape[1]
        
        if LFR_mitigation==True:
            n_feat=len(X_col_names)-1#minus gender
        
        else:
            n_feat=len(X_col_names)
        
        
        output_dim=1 #binary


        data_tr = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=False)
        data_val = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False)
        data_ts = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)




        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device="cpu"
        print(device)


        model = Network().to(device)



        opt=optim.Adam(model.parameters(),lr=lr, weight_decay = wd)
        

        epochnumber = []
        all_train_losses = []
        all_val_losses = []
        all_ts_losses = []

        all_train_acc=[]
        all_val_acc=[]
        all_ts_acc=[]

        all_time_low_train_loss=1000
        all_time_low_val_loss=1000

        all_time_high_train_acc=0
        all_time_high_val_acc=0


        for epoch in range(epochs):
            if (epoch)%20==0:
                print('* Epoch %d/%d' % (epoch+1, epochs))

            epochnumber.append(epoch)

            avg_loss_train = 0
            avg_acc=0
            model.train()  # train mode
            for X_batch, Y_batch in data_tr:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)



                # set parameter gradients to zero
                opt.zero_grad()

                # forward
                Y_pred = model(X_batch.float()) #oprdindeligt havde vi 3 lag (RGB), nu har vi kun 1 (greyscale) -> 
                loss = loss_fn(Y_batch.float(), Y_pred.squeeze())  # forward-pass
                loss.backward()  # backward-pass
                opt.step()  # update weights

                # calculate metrics to show the user
                avg_loss_train += loss / len(data_tr)

                avg_acc+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_tr)


            all_time_low_train_loss=get_all_time_low(all_time_low_train_loss,avg_loss_train)
            all_time_low_train_acc=get_all_time_high(all_time_high_train_acc,avg_acc)
              #print(' - train loss: %f' % avg_loss_train)
              #print(' - train acc: {} %'.format(round(avg_acc,2)))

            all_train_losses.append(avg_loss_train)
            all_train_acc.append(avg_acc)



            with torch.no_grad():
                avg_loss_val = 0
                avg_acc_val=0
                model.eval()  # eval mode
                for X_batch, Y_batch in data_val:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)


                    # forward
                    Y_pred = model(X_batch.float()) 
                    loss = loss_fn(Y_batch.float(), Y_pred.squeeze())  # forward-pass

                    # calculate metrics to show the user
                    avg_loss_val += loss / len(data_val)
                    avg_acc_val+=accuracy(Y_batch,Y_pred.squeeze()) / len(data_val)
                #toc = time()
                all_time_low_val_loss=get_all_time_low(all_time_low_val_loss,avg_loss_val)
                all_time_low_val_acc=get_all_time_high(all_time_high_val_acc,avg_acc_val)
                #print(' - val loss: %f' % avg_loss_val)
                #print(' - val acc: {} %'.format(round(avg_acc_val,2)))


                ########Save model####

            if  epoch == 0 or avg_loss_val <= min(all_val_losses) :
                torch.save(model.state_dict(), PATH+'_FFNN_model_local.pth')
                print('####Saved model####')

            all_val_losses.append(avg_loss_val)
            all_val_acc.append(avg_acc_val)




          ###PLOT########

        if epoch==epochs-1:
            #Save the last epoch
            torch.save(model.state_dict(), PATH+'_FFNN_model_global.pth')

            #take the best model (with lowest validation loss)
            model.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))
            model.eval()

            all_ts_losses=[get_test()[0]] * (epoch+1)
            all_ts_acc=[get_test()[1]] * (epoch+1)

            plt.figure(1)
            plt.plot(epochnumber, all_train_losses, 'r', epochnumber, all_val_losses, 'b',epochnumber, all_ts_losses, '--')
            plt.xlabel('Epochs'), plt.ylabel('Loss')
            plt.legend(['Train Loss', 'Val Loss','Test loss'])
            plt.savefig(PATH+'_loss.png')
            plt.show()

            plt.figure(2)
            plt.plot(epochnumber, all_train_acc, 'black', epochnumber, all_val_acc, 'grey',epochnumber, all_ts_acc, '--')
            plt.xlabel('Epochs'), plt.ylabel('Accuracy')
            plt.legend(['Train acc', 'Val acc','Test acc'])
            plt.savefig(PATH+'_acc.png')
            plt.show()



            metrics=pd.DataFrame({"all_time_low_train_loss":[all_time_low_train_loss.item()],
                                  "all_time_low_train_acc":[all_time_low_train_acc],
                              "all_time_low_val_loss":[all_time_low_val_loss.item()],
                                  "all_time_val_train_acc":[all_time_low_val_acc],
                              "test_acc":[all_ts_acc[0]],
                              "test_loss":[all_ts_losses[0].item()]
                                                })
            metrics.to_csv( PATH+'_metrics.csv')





            for local_best in [0,1]:
                #local_best=0

                model1 = Network().to(device)
                if local_best==1:
                    model1.load_state_dict(torch.load(PATH+'_FFNN_model_local.pth'))
                else:
                    model1.load_state_dict(torch.load(PATH+'_FFNN_model_global.pth'))

                model1.eval()


                df_evaluate = X_test_pd.copy()
                df_evaluate[y_col_name]=y_test_pd[y_col_name]

                if dropping_D==True:
                    df_evaluate[procted_col_name]=y_test_pd[procted_col_name]

                else:
                    df_evaluate[procted_col_name]=X_test_pd[procted_col_name]

                if AIR==False:
                    cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name,"sex","age_cat","race","c_charge_desc","c_charge_degree"]]
                else:
                    if dropping_D==True:
                        cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name,procted_col_name]]

                    elif gender_swap==True:
                        cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name,"Original"]]
                    else:
                        cols= [col for col in list(df_evaluate.columns) if col not in [y_col_name]]




                X_numpy=np.array(df_evaluate[cols])
                X_torch=torch.tensor(X_numpy)
                y_pred = model1(X_torch.float().to(device))


                list_of_output=[round(a.item(),0) for a in y_pred.detach().cpu()]
                list_of_output_prob=[a.item() for a in y_pred.detach().cpu()]

                df_evaluate["output"]=list_of_output
                df_evaluate["output_prob"]=list_of_output_prob
                df_evaluate["Model"]=seedName


                ##SAVING THE TEST DATA
                if local_best==1:
                    df_evaluate.to_csv(PATH+"test_data_localmodel.csv")
                else:
                    df_evaluate.to_csv(PATH+"test_data_globalmodel.csv")
            

        modelcounter=modelcounter+1


# # Save all test data (and output)

# In[13]:




for file_name in ["localmodel","globalmodel"]:
    first_time=True
    
    for j in range(modelcounter):
        
        if first_time==True:
            test_data_all = pd.read_csv(PATH_orig+"model"+str(j)+"/test_data_"+file_name+".csv")
            first_time=False
        else:
            test_subset=pd.read_csv(PATH_orig+"model"+str(j)+"/test_data_"+file_name+".csv")
            test_data_all=pd.concat([test_data_all,test_subset],sort=False,axis=0)
       

       

    test_data_all.to_csv(PATH_orig+"all_test_data_"+file_name+".csv")
    print(f"the shape of {file_name} is {test_data_all.shape}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#  # GLOBAL ALL

# In[14]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):
  
#    PATH_loop=PATH_orig+"model"+str(i)+"/_all_stats_global.csv"
  
#    data=pd.read_csv(PATH_loop)
#    for group in ["all"]:
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP']:
#            value=float(data[data[procted_col_name]==group][measure])

#            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH_orig+"/FFNN_metrics_crossvalidated_global_all.csv")


# In[15]:


#global_all_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#global_all_bar.set_title('Global all')
#global_all_bar.get_figure().savefig(PATH_orig+"/barplot_global_all.png")


# # LOCAL ALL

# In[16]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):
#
#    PATH_loop=PATH_orig+"model"+str(i)+"/_all_stats_local.csv"
#  
#    data=pd.read_csv(PATH_loop)
#    for group in ["all"]:
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP',"y_hat_mean","y_target_mean"]:
#            value=float(data[data[procted_col_name]==group][measure])

#            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_local_all.csv")


# In[17]:


#local_all_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#local_all_bar.set_title('Global all')
#local_all_bar.get_figure().savefig(PATH_orig+"/barplot_local_all.png")


# # Global protected

# In[18]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):

#    PATH_loop=PATH_orig+"model"+str(i)+"/_"+procted_col_name+"_stats_global.csv"
  
#    data=pd.read_csv(PATH_loop)
#    for group in list(data[procted_col_name].unique()):
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP',"y_hat_mean","y_target_mean"]:
#            value=float(data[data[procted_col_name]==group][measure])

 #           df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_global_"+procted_col_name+".csv")


# In[19]:


#global_proc_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#global_proc_bar.set_title('Global proctected: '+procted_col_name)
#global_proc_bar.get_figure().savefig(PATH_orig+"/barplot_global_proc.png")


# # Local protected

# In[20]:


#column_names = ["Group", "ML", "Measure","Value"]

#df_out = pd.DataFrame(columns = column_names)

#for i in range(50):
#    PATH_loop=PATH_orig+"model"+str(i)+"/_"+procted_col_name+"_stats_local.csv"
  
#    data=pd.read_csv(PATH_loop)
#    for group in list(data[procted_col_name].unique()):
#        for measure in ['FPR', 'FNR', 'ACC', 'F1', 'FDR', 'LRminus','LRplus', 'NPV', 'PPV', 'TNR', 'TPR','TP','TN','FN','FP',"y_hat_mean","y_target_mean"]:
#            value=float(data[data[procted_col_name]==group][measure])

#            df_out=df_out.append({'Group': group,"ML":"FFNN"+str(i),"Measure":measure,"Value":value}, ignore_index=True)

#df_out.to_csv(PATH_orig+"FFNN_metrics_crossvalidated_local_"+procted_col_name+".csv")


# In[21]:


#local_proc_bar=sns.barplot(data=df_out[df_out["Measure"].isin(["FPR","FNR","TPR","TNR"])],x="Group", y="Value", ci=95,hue="Measure")
#local_proc_bar.set_title('Local protected: '+procted_col_name)
#local_proc_bar.get_figure().savefig(PATH_orig+"/barplot_local_proc.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:



'''
for file_name in ["localmodel","globalmodel"]:
    
    for j in range(modelcounter):
    
        test_data_0 = pd.read_csv(PATH_orig+"model0/test_data_"+file_name+".csv")
        test_data_1 = pd.read_csv(PATH_orig+"model1/test_data_"+file_name+".csv")
        test_data_2 = pd.read_csv(PATH_orig+"model2/test_data_"+file_name+".csv")
        test_data_3 = pd.read_csv(PATH_orig+"model3/test_data_"+file_name+".csv")
        test_data_4 = pd.read_csv(PATH_orig+"model4/test_data_"+file_name+".csv")
        test_data_5 = pd.read_csv(PATH_orig+"model5/test_data_"+file_name+".csv")
        test_data_6 = pd.read_csv(PATH_orig+"model6/test_data_"+file_name+".csv")
        test_data_7 = pd.read_csv(PATH_orig+"model7/test_data_"+file_name+".csv")
        test_data_8 = pd.read_csv(PATH_orig+"model8/test_data_"+file_name+".csv")
        test_data_9 = pd.read_csv(PATH_orig+"model9/test_data_"+file_name+".csv")

        df2=    pd.concat([test_data_0,
                            test_data_1,
                            test_data_2,
                            test_data_3,
                            test_data_4,
                            test_data_5,
                            test_data_6,
                            test_data_7,
                            test_data_8,
                            test_data_9
                           ],sort=False,axis=0)

    df2.to_csv(PATH_orig+"all_test_data_"+file_name+".csv")
'''


# In[ ]:




