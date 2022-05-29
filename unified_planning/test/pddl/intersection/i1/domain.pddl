(define (domain intersection)
 
 (:requirements :strips :typing)
 
 (:types
  direction loc car - object
        
 )

 (:predicates 
  (at ?a - car ?l - loc)  
  (free ?l - loc)  
  
  (arrived ?a - car)
  (start ?a - car ?l - loc)
  (travel-direction ?a - car ?d - direction)
  
  (connected ?l1 - loc ?l2 - loc ?d - direction)
  
      
 )
 
 (:action arrive
    :parameters  (?a - car  ?l - loc)
    :precondition  (and  
    	(start ?a ?l)
    	(not (arrived ?a))
    	(free ?l)      
      )
    :effect    (and     	
    	(at ?a ?l)
    	(not (free ?l))
    	(arrived ?a)
      )
  )
  
  (:action drive
    :parameters  (?a - car  ?l1 - loc ?l2 - loc ?d - direction)
    :precondition  (and      	
    	(at ?a ?l1)
    	(free ?l2)     
    	(travel-direction ?a ?d)
    	(connected ?l1 ?l2 ?d)
      )
    :effect    (and     	
    	(at ?a ?l2)
    	(not (free ?l2))
    	(not (at ?a ?l1))
    	(free ?l1)
      )
  )    
)


