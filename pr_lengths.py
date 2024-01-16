def protocol_length(Pr=7):
    if Pr == 7 or Pr == 0:  #PR7 and PRX
        return 80000
    elif Pr == 6:
        return 88245
    elif Pr == 5:
        return 929016
    elif Pr == 4:
        return 464096
    elif Pr == 3:
        return 578060
    elif Pr == 2:
        return 312000
    elif Pr == 1:
        return 312000
    else:
        print('An error has occurred-- the PR length function was not given a valid protocol (int 1-7')

if __name__ == '__main__':
    prl = protocol_length(1)
    print('Protocol 1 has a length of '+str(prl))
    prl = protocol_length(2)                     
    print('Protocol 2 has a length of '+str(prl))
    prl = protocol_length(3)
    print('Protocol 3 has a length of '+str(prl))
    prl = protocol_length(4)
    print('Protocol 4 has a length of '+str(prl))
    prl = protocol_length(5)
    print('Protocol 5 has a length of '+str(prl))
    prl = protocol_length(6)
    print('Protocol 6 has a length of '+str(prl))
    prl = protocol_length(7)
    print('Protocol 7 has a length of '+str(prl))
