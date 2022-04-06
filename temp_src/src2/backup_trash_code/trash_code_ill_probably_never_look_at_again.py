            """
            #Expression A: Running circuits produced above,  this can most certainly be done in parallel
            counter=0
            for i in range(len(self.rot_indexes)):
                for j in range(len(self.rot_indexes)):
                    #A_mat[i][j]=matrix_values[counter]*0.25
                    A_mat[i][j]=run_circuit(qc_list_A[counter])
                    counter+=1
            A_mat*=0.25
            """
            #Expression C: Binds the parameters to the circuits
            """
            counter_C=0
            for i in range(len(self.hamil)):
                for j in range(len(self.rot_indexes)):
                    #Just the circuits
                    qc_list_C.append(self.C_init[i][j].bind_parameters(labels[len(self.rot_indexes):len(self.C_init[i][j].parameters):-1]))
                    counter_C+=1
                    #n_rotations=len(self.A_init[i][j].parameters)
                    #print('-------------')
                    #print(self.A_init[i][j])
                    #print(len(self.A_init[i][j].parameters), i, j)
            
            
                    
                    They produces the same circuits
                    trash, circ=self.run_A2(self.rot_indexes[i],self.rot_indexes[j])
                    print(i,j,Statevector.from_instruction(self.A_init[i][j].bind_parameters(\
                    labels[:len(self.A_init[i][j].parameters)])).equiv(Statevector.from_instruction(circ)))
 
            #Expression C: Running circuits produced above
            counter2_C=0
            for i in range(len(self.hamil)):
                for j in range(len(self.rot_indexes)):
                    #A_mat[i][j]=matrix_values[counter]*0.25
                    C_vec[j]+=run_circuit(qc_list_C[counter2_C])
                    counter2_C+=1
            C_vec*=0.5
            """


               """
            if i want to use this:   
                #A_mat_test=(self.A_init)
                #C_vec_test=self.C_init.copy()
                #Remember to multiply with (0+0.5j)*(0-0.5j)
                circ=[]
                start_loop=time.time()
                for ii in range(len(A_mat_test)):
                    for jj in range(len(A_mat_test[0])):
                        circ_test=A_mat_test[ii][jj]
                        
                        if circ_test!=None:
                            n_rotations=len(circ_test.parameters)
                            circ_test=circ_test.bind_parameters(labels[:n_rotations])
                            circ.append(circ_test)
                            circ_pred=run_circuit(circ_test)
                            A_mat_test[ii][jj]=circ_pred*0.25
                        else:
                            A_mat_test[ii][jj]=0.
                end_loop=time.time()
                #print(f'old mat {end_mat-start_mat}')
                #print(f'loop {end_loop-start_loop}')


                circ_pred=0

                for ii in range(len(C_vec_test)):
                    circ_test=C_vec_test[ii]
                    gate=self.trial_circ[ii][0]
                    if gate== 'cx' or gate == 'cy' or gate == 'cz':
                        C_vec_test[ii]=0
                    else:
                        for jj in range(len(C_vec_test[ii])):
        
                            temp_list=[]

                            n_rotations=len(circ_test[jj].parameters)
                            circ_test[jj]=circ_test[jj].bind_parameters(labels[:n_rotations])
                            #Just appending for option to run paralell circuits
                            temp_list.append(circ_test[jj])
                            #print(f'lambda is {self.hamil[:0][self.C_lmb_index[ii]]}')
                            circ_pred=run_circuit(circ_test[jj])

                            C_vec_test[ii]=circ_pred*0.5*lmbs[self.C_lmb_index[ii][jj]]
                        #else:
                                #C_vec_test[ii]=0.
                            
                            circ.append(temp_list)

                Just do it the old way, make it work and optimize later
                #print(A_mat2)
                print(C_vec2)
                C_vec2_2=np.zeros(len(self.trial_circ))
                C_vec2_2[self.rot_indexes]=C_vec_test
                print(C_vec2)
                print(C_vec2_2)
                print(np.all(C_vec2==C_vec2_2))     

                C_vec2=C_vec2_2
            """