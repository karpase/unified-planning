(define (problem intersection-01)

	(:domain intersection)
	
	(:objects
		a1 a2 a3 a4 - car	
		south-ent south-ex north-ent north-ex east-ent  east-ex west-ent  west-ex cross-nw cross-ne cross-se cross-sw - loc
   		north south east west - direction	
	)
		
	(:init
		(start a1 south-ent)
		(start a2 north-ent)
		(start a3 east-ent)		
		(start a4 west-ent)
		(travel-direction a1 north)
		(travel-direction a2 south)
		(travel-direction a3 west)
		(travel-direction a4 east)
		
		(connected south-ent cross-se north)
		(connected cross-se cross-ne north)
		(connected cross-ne north-ex north)
		
		(connected north-ent cross-nw south)
		(connected cross-nw cross-sw south)
		(connected cross-sw south-ex south)
		
		(connected east-ent cross-ne west)
		(connected cross-ne cross-nw west)
		(connected cross-nw west-ex west)
		
		(connected west-ent cross-sw east)
		(connected cross-sw cross-se east)
		(connected cross-se east-ex east)
		
		
		(free north-ent)
		(free south-ent)
		(free west-ent)
		(free east-ent)		
		(free north-ex)
		(free south-ex)
		(free west-ex)
		(free east-ex)
		(free cross-ne) (free cross-nw) (free cross-se) (free cross-sw)
		
		
	)
	
	(:goal 	
		(and	
			(at a1 north-ex)
			(at a2 south-ex)
			(at a3 west-ex)
			(at a4 east-ex)
		)
	)
)
